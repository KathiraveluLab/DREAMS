"""The stale-claim cleanup and the claim itself must use the same clock."""

import datetime
import time
from unittest.mock import MagicMock, patch

import pytest

from dreams_app.app import fl_worker


@pytest.fixture
def server_in_alaska(monkeypatch):
    """Run with the server's local time zone set to UTC-8/-9, as it would be in Anchorage."""
    monkeypatch.setenv('TZ', 'America/Anchorage')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _run_round_until_claim():
    """Run one round up to the claim step and return the mocked posts collection."""
    posts = MagicMock()
    posts.find.return_value.limit.return_value = [{'_id': i} for i in range(50)]
    # Stale cleanup resets nothing; the claim then reports that nothing was claimed,
    # which makes the round release its candidates and stop.
    posts.update_many.return_value = MagicMock(modified_count=0)

    app = MagicMock()
    app.config = {}
    app.mongo.__getitem__.return_value = posts

    with patch.object(fl_worker, 'torch', MagicMock()), \
         patch.object(fl_worker, 'AutoModelForSequenceClassification', MagicMock()), \
         patch.object(fl_worker, 'AutoTokenizer', MagicMock()), \
         patch.object(fl_worker, 'AutoConfig', MagicMock()):
        fl_worker.run_federated_round(app)
    return posts


def test_a_batch_claimed_now_is_not_treated_as_stale(server_in_alaska):
    posts = _run_round_until_claim()

    stale_filter = posts.update_many.call_args_list[0][0][0]
    claim_update = posts.update_many.call_args_list[1][0][1]
    cutoff = stale_filter['processing_started_at']['$lt']
    claimed_at = claim_update['$set']['processing_started_at']

    # The cutoff is one hour ago, so a claim made now has to be later than it.
    assert claimed_at > cutoff
    assert claimed_at - cutoff == pytest.approx(datetime.timedelta(hours=1), abs=datetime.timedelta(seconds=5))
