"""Tests for DreamsPipeline.generate_narrative_metrics with inconsistent stored posts."""

from datetime import datetime, timedelta, timezone

from dreams_app.core.pipeline import DreamsPipeline

BASE = datetime(2026, 1, 1, 10, 0)


def _post(post_id, hours, label="positive", **overrides):
    post = {
        "_id": post_id,
        "timestamp": BASE + timedelta(hours=hours),
        "sentiment": {"label": label, "score": 0.9},
    }
    post.update(overrides)
    return post


def _metrics(posts):
    return DreamsPipeline().generate_narrative_metrics("user-1", posts)


def test_naive_and_aware_timestamps_can_be_mixed():
    mixed = [
        _post(1, 0),
        _post(2, 1, "negative", timestamp=(BASE + timedelta(hours=1)).replace(tzinfo=timezone.utc)),
        _post(3, 2, timestamp="2026-01-01T12:00:00+00:00"),
    ]
    all_naive = [_post(1, 0), _post(2, 1, "negative"), _post(3, 2)]

    assert _metrics(mixed) == _metrics(all_naive)


def test_an_aware_timestamp_gives_the_same_result_as_the_same_naive_utc_one():
    naive = [_post(1, 0), _post(2, 1, "negative"), _post(3, 2)]
    aware = [dict(p, timestamp=p["timestamp"].replace(tzinfo=timezone.utc)) for p in naive]

    assert _metrics(aware) == _metrics(naive)


def test_a_non_utc_offset_is_converted_to_utc():
    # 05:00 at UTC-5 is 10:00 UTC, the same instant as BASE.
    minus_five = timezone(timedelta(hours=-5))
    same_instant = dict(_post(1, 0), timestamp=datetime(2026, 1, 1, 5, 0, tzinfo=minus_five))

    assert _metrics([same_instant, _post(2, 1)]) == _metrics([_post(1, 0), _post(2, 1)])


def test_a_post_with_null_sentiment_is_handled_like_one_without_sentiment():
    null_sentiment = [_post(1, 0, sentiment=None), _post(2, 1)]
    empty_sentiment = [_post(1, 0, sentiment={}), _post(2, 1)]

    assert _metrics(null_sentiment) == _metrics(empty_sentiment)
