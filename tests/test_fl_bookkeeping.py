"""Bookkeeping tests for ``fl_worker.run_federated_round``.

The ML stack is replaced by small fakes so the tests exercise only how a round
claims, finishes and releases correction documents in MongoDB.
"""
import datetime
import os
import time
from types import SimpleNamespace

import pytest

mongomock = pytest.importorskip("mongomock")

from dreams_app.app import fl_worker

LABELS = {"Connectedness": 0, "Hope": 1, "Identity": 2, "Meaning": 3, "Empowerment": 4}
BATCH = 4


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


class FakeLoss:
    def backward(self):
        pass

    def item(self):
        return 0.0


class FakeParams:
    def parameters(self):
        return []


class FakeModel:
    base_model_prefix = "bert"

    def __init__(self, on_forward=None):
        self.bert = FakeParams()
        self.classifier = FakeParams()
        self.on_forward = on_forward

    def train(self):
        pass

    def __call__(self, **kwargs):
        if self.on_forward:
            self.on_forward()
        return SimpleNamespace(loss=FakeLoss())

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


class FakeTokenizer:
    def __call__(self, *args, **kwargs):
        return {}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


class FakeOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """Patch fl_worker with a fake ML stack; ``state['on_forward']`` runs mid-training."""
    state = {"on_forward": None, "passed": True}
    fake_torch = SimpleNamespace(
        tensor=lambda values: values,
        optim=SimpleNamespace(AdamW=FakeOptimizer),
    )
    monkeypatch.setattr(fl_worker, "torch", fake_torch)
    monkeypatch.setattr(fl_worker, "AutoConfig", SimpleNamespace(from_pretrained=lambda _id: SimpleNamespace(label2id=LABELS)))
    monkeypatch.setattr(fl_worker, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda _p: FakeTokenizer()))
    monkeypatch.setattr(
        fl_worker,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=lambda _p, num_labels: FakeModel(lambda: state["on_forward"] and state["on_forward"]())),
    )
    monkeypatch.setattr(fl_worker, "validate_model", lambda *a, **k: state["passed"])
    monkeypatch.setattr(fl_worker, "PRODUCTION_MODEL_DIR", str(tmp_path / "production"))
    monkeypatch.setattr(fl_worker, "TEMP_MODEL_DIR", str(tmp_path / "temp"))
    return state


@pytest.fixture
def app():
    app = SimpleNamespace(
        mongo=mongomock.MongoClient()["dreams_test"],
        config={"FL_BATCH_SIZE": BATCH},
    )
    app.app_context = lambda: _NullContext()
    return app


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _add_corrections(posts, count=BATCH):
    return posts.insert_many(
        [{"caption": f"caption {i}", "corrected_label": "Hope", "is_fl_processed": False} for i in range(count)]
    ).inserted_ids


def test_finished_round_marks_claimed_posts_trained(worker, app):
    posts = app.mongo["posts"]
    ids = _add_corrections(posts)

    fl_worker.run_federated_round(app=app)

    docs = list(posts.find({"_id": {"$in": ids}}))
    assert all(d["is_fl_processed"] is True and d["fl_status"] == "trained" for d in docs)
    assert not any("fl_round_id" in d or "processing_started_at" in d for d in docs)


def test_rejected_round_is_recorded_on_the_posts(worker, app):
    worker["passed"] = False
    posts = app.mongo["posts"]
    ids = _add_corrections(posts)

    fl_worker.run_federated_round(app=app)

    assert {d["fl_status"] for d in posts.find({"_id": {"$in": ids}})} == {"rejected"}


def test_correction_made_during_training_stays_pending(worker, app):
    posts = app.mongo["posts"]
    ids = _add_corrections(posts)

    def user_recorrects_post():
        # Same update the /correct_chime route performs while a round is running.
        posts.update_one({"_id": ids[0]}, {"$set": {"corrected_label": "Meaning", "is_fl_processed": False}})
        worker["on_forward"] = None

    worker["on_forward"] = user_recorrects_post

    fl_worker.run_federated_round(app=app)

    recorrected = posts.find_one({"_id": ids[0]})
    assert recorrected["corrected_label"] == "Meaning"
    assert recorrected["is_fl_processed"] is False
    assert posts.count_documents({"_id": {"$in": ids[1:]}, "is_fl_processed": True}) == BATCH - 1


def test_failed_round_leaves_other_workers_claims_alone(worker, app, monkeypatch):
    posts = app.mongo["posts"]
    other = posts.insert_one(
        {
            "caption": "claimed elsewhere",
            "corrected_label": "Hope",
            "is_fl_processed": "processing",
            "fl_round_id": "another-worker",
            "processing_started_at": _utcnow(),
        }
    ).inserted_id
    ids = _add_corrections(posts)

    def boom():
        raise RuntimeError("training crashed")

    worker["on_forward"] = boom

    fl_worker.run_federated_round(app=app)

    assert posts.find_one({"_id": other})["is_fl_processed"] == "processing"
    assert posts.count_documents({"_id": {"$in": ids}, "is_fl_processed": False}) == BATCH


def test_claim_timestamp_is_utc_whatever_the_server_timezone(worker, app, monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("tzset is not available on this platform")
    monkeypatch.setenv("TZ", "America/Anchorage")
    time.tzset()
    try:
        posts = app.mongo["posts"]
        ids = _add_corrections(posts)
        seen = {}

        def capture_claim():
            seen["started"] = posts.find_one({"_id": ids[0]})["processing_started_at"]
            worker["on_forward"] = None

        worker["on_forward"] = capture_claim

        fl_worker.run_federated_round(app=app)

        started = seen["started"]
        if started.tzinfo is None:
            started = started.replace(tzinfo=datetime.timezone.utc)
        assert abs((_utcnow() - started).total_seconds()) < 60
    finally:
        monkeypatch.undo()
        time.tzset()


def test_stale_claims_are_released_even_in_a_negative_utc_offset(worker, app, monkeypatch):
    """A claim made seconds ago must not look an hour old to the stale-claim cleanup."""
    if not hasattr(time, "tzset"):
        pytest.skip("tzset is not available on this platform")
    monkeypatch.setenv("TZ", "America/Anchorage")
    time.tzset()
    try:
        posts = app.mongo["posts"]
        fresh = posts.insert_one(
            {
                "caption": "just claimed",
                "corrected_label": "Hope",
                "is_fl_processed": "processing",
                "fl_round_id": "another-worker",
                "processing_started_at": _utcnow() - datetime.timedelta(minutes=2),
            }
        ).inserted_id
        stale = posts.insert_one(
            {
                "caption": "abandoned",
                "corrected_label": "Hope",
                "is_fl_processed": "processing",
                "fl_round_id": "crashed-worker",
                "processing_started_at": _utcnow() - datetime.timedelta(hours=2),
            }
        ).inserted_id

        fl_worker.run_federated_round(app=app)  # fewer than BATCH pending, so it only cleans up

        assert posts.find_one({"_id": fresh})["is_fl_processed"] == "processing"
        assert posts.find_one({"_id": stale})["is_fl_processed"] is False
    finally:
        monkeypatch.undo()
        time.tzset()
