"""Tests for the analytics demo server's sample data and user handling."""
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")

import analytics_server  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SNIPPET = (
    "import analytics_server as a; "
    "posts = a.generate_sample_posts('{user}'); "
    "print(len(posts), posts[0]['timestamp'].isoformat(), posts[0]['sentiment']['score'], "
    "posts[-1]['timestamp'].isoformat())"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from dreams_app.core.graph.persistence import ContentAddressedStore, StructuralCache

    store = ContentAddressedStore(tmp_path)
    monkeypatch.setattr(analytics_server, "cache_dir", tmp_path)
    monkeypatch.setattr(analytics_server, "store", store)
    monkeypatch.setattr(analytics_server, "cache", StructuralCache(store))
    return analytics_server.app.test_client()


def _sample_in_new_process(user, hash_seed):
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET.format(user=user)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


# --- sample data is the same in every process ------------------------------------------------


@pytest.mark.parametrize("user", ["user_001", "user_002", "user_003"])
def test_sample_posts_do_not_depend_on_the_process_hash_seed(user):
    outputs = {_sample_in_new_process(user, seed) for seed in (0, 1, 2, 3)}

    assert len(outputs) == 1


def test_sample_posts_are_repeatable_within_a_process():
    first = analytics_server.generate_sample_posts("user_003")
    second = analytics_server.generate_sample_posts("user_003")

    assert first == second


def test_users_get_different_sample_posts():
    assert analytics_server.generate_sample_posts("user_001") != analytics_server.generate_sample_posts(
        "user_003"
    )


def test_generating_sample_posts_leaves_the_global_random_state_alone():
    random.seed(1234)
    expected = random.getstate()

    analytics_server.generate_sample_posts("user_001")

    assert random.getstate() == expected


# --- only the demo users have data ----------------------------------------------------------

API_ROUTES = [
    "/api/timeline/{}",
    "/api/episodes/{}",
    "/api/narrative-graph/{}",
    "/api/frontend-payload/{}",
    "/api/cache-status/{}",
    "/narrative/{}",
]


@pytest.mark.parametrize("route", API_ROUTES)
def test_unknown_user_gets_404(client, route):
    response = client.get(route.format("not-a-user"))

    assert response.status_code == 404
    assert response.get_json()["error"] == "User not found"


@pytest.mark.parametrize("route", API_ROUTES)
@pytest.mark.parametrize("user", ["user_001", "user_002", "user_003"])
def test_demo_users_still_work(client, route, user):
    assert client.get(route.format(user)).status_code == 200


def test_made_up_ids_do_not_fill_the_cache_directory(client, tmp_path):
    for i in range(20):
        client.get(f"/api/timeline/made-up-{i}")
        client.get(f"/api/narrative-graph/made-up-{i}")

    assert list(tmp_path.iterdir()) == []


def test_demo_users_use_a_bounded_number_of_cache_files(client, tmp_path):
    for _ in range(3):
        for user in analytics_server.SAMPLE_USERS:
            client.get(f"/api/timeline/{user}")
            client.get(f"/api/narrative-graph/{user}")

    assert len(list(tmp_path.iterdir())) <= 2 * len(analytics_server.SAMPLE_USERS)
