# tests/test_time_aware_proximity.py

"""
Tests for time-aware proximity and comparison layer.

Covers:
- Empty timelines
- Single-event timelines
- Unequal lengths
- Sparse timelines
- Deterministic output
"""

import pytest
from datetime import datetime, timedelta

from dreamsApp.analytics.emotion_timeline import EmotionTimeline, EmotionEvent
from dreamsApp.analytics.time_aware_proximity import (
    align_timelines_by_window,
    temporal_distance,
    proximity_matrix,
)


# Fixtures

@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def empty_timeline() -> EmotionTimeline:
    return EmotionTimeline(subject_id="empty", events=())


@pytest.fixture
def single_event_timeline(base_time: datetime) -> EmotionTimeline:
    event = EmotionEvent(timestamp=base_time, emotion_label="neutral")
    return EmotionTimeline(subject_id="single", events=(event,))


@pytest.fixture
def multi_event_timeline(base_time: datetime) -> EmotionTimeline:
    events = (
        EmotionEvent(timestamp=base_time, emotion_label="positive"),
        EmotionEvent(timestamp=base_time + timedelta(hours=1), emotion_label="neutral"),
        EmotionEvent(timestamp=base_time + timedelta(hours=2), emotion_label="negative"),
    )
    return EmotionTimeline(subject_id="multi", events=events)


@pytest.fixture
def sparse_timeline(base_time: datetime) -> EmotionTimeline:
    events = (
        EmotionEvent(timestamp=base_time, emotion_label="positive"),
        EmotionEvent(timestamp=base_time + timedelta(hours=5), emotion_label="negative"),
    )
    return EmotionTimeline(subject_id="sparse", events=events)


@pytest.fixture
def dense_timeline(base_time: datetime) -> EmotionTimeline:
    events = tuple(
        EmotionEvent(
            timestamp=base_time + timedelta(minutes=30 * i),
            emotion_label="neutral"
        )
        for i in range(6)
    )
    return EmotionTimeline(subject_id="dense", events=events)


# Tests for align_timelines_by_window

class TestAlignTimelinesByWindow:
    
    def test_empty_timelines_tuple(self):
        result = align_timelines_by_window(
            timelines=(),
            window=timedelta(hours=1),
            anchor="start"
        )
        assert result == {}
    
    def test_all_empty_timelines(self, empty_timeline: EmotionTimeline):
        result = align_timelines_by_window(
            timelines=(empty_timeline, empty_timeline),
            window=timedelta(hours=1),
            anchor="start"
        )
        assert result == {}
    
    def test_single_timeline_single_event(
        self,
        single_event_timeline: EmotionTimeline
    ):
        result = align_timelines_by_window(
            timelines=(single_event_timeline,),
            window=timedelta(hours=1),
            anchor="start"
        )
        assert len(result) >= 1
        assert result[0][0] is not None
        assert result[0][0].emotion_label == "neutral"
    
    def test_multiple_timelines_alignment(
        self,
        base_time: datetime,
        multi_event_timeline: EmotionTimeline
    ):
        events2 = (
            EmotionEvent(timestamp=base_time + timedelta(minutes=30), emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(hours=1, minutes=30), emotion_label="positive"),
        )
        timeline2 = EmotionTimeline(subject_id="other", events=events2)
        
        result = align_timelines_by_window(
            timelines=(multi_event_timeline, timeline2),
            window=timedelta(hours=1),
            anchor="start"
        )
        
        assert len(result) >= 1
        for window_idx, events in result.items():
            assert len(events) == 2
    
    def test_anchor_start(
        self,
        base_time: datetime,
        multi_event_timeline: EmotionTimeline
    ):
        result = align_timelines_by_window(
            timelines=(multi_event_timeline,),
            window=timedelta(hours=1),
            anchor="start"
        )
        
        assert 0 in result
        assert result[0][0] is not None
    
    def test_anchor_end(
        self,
        base_time: datetime,
        multi_event_timeline: EmotionTimeline
    ):
        result = align_timelines_by_window(
            timelines=(multi_event_timeline,),
            window=timedelta(hours=1),
            anchor="end"
        )
        
        assert len(result) >= 1
    
    def test_anchor_explicit(
        self,
        base_time: datetime,
        multi_event_timeline: EmotionTimeline
    ):
        result = align_timelines_by_window(
            timelines=(multi_event_timeline,),
            window=timedelta(hours=1),
            anchor="explicit",
            anchor_time=base_time - timedelta(hours=1)
        )
        
        assert len(result) >= 1
    
    def test_explicit_anchor_requires_time(
        self,
        multi_event_timeline: EmotionTimeline
    ):
        with pytest.raises(ValueError, match="anchor_time required"):
            align_timelines_by_window(
                timelines=(multi_event_timeline,),
                window=timedelta(hours=1),
                anchor="explicit",
                anchor_time=None
            )
    
    def test_invalid_window(self, multi_event_timeline: EmotionTimeline):
        with pytest.raises(ValueError, match="positive timedelta"):
            align_timelines_by_window(
                timelines=(multi_event_timeline,),
                window=timedelta(0),
                anchor="start"
            )
    
    def test_deterministic_output(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline
    ):
        result1 = align_timelines_by_window(
            timelines=(multi_event_timeline, sparse_timeline),
            window=timedelta(hours=1),
            anchor="start"
        )
        result2 = align_timelines_by_window(
            timelines=(multi_event_timeline, sparse_timeline),
            window=timedelta(hours=1),
            anchor="start"
        )
        
        assert result1 == result2
    
    def test_sparse_timeline_has_none_values(
        self,
        sparse_timeline: EmotionTimeline
    ):
        result = align_timelines_by_window(
            timelines=(sparse_timeline,),
            window=timedelta(hours=1),
            anchor="start"
        )
        
        none_count = sum(1 for events in result.values() if events[0] is None)
        assert none_count > 0
    
    def test_at_most_one_event_per_window(
        self,
        dense_timeline: EmotionTimeline
    ):
        result = align_timelines_by_window(
            timelines=(dense_timeline,),
            window=timedelta(hours=1),
            anchor="start"
        )
        
        for window_idx, events in result.items():
            assert len(events) == 1


# Tests for temporal_distance

class TestTemporalDistance:
    
    def test_both_empty(self, empty_timeline: EmotionTimeline):
        dist = temporal_distance(
            empty_timeline,
            empty_timeline,
            window=timedelta(hours=1)
        )
        assert dist == 0.0
    
    def test_one_empty_one_non_empty(
        self,
        empty_timeline: EmotionTimeline,
        single_event_timeline: EmotionTimeline
    ):
        dist = temporal_distance(
            empty_timeline,
            single_event_timeline,
            window=timedelta(hours=1)
        )
        assert dist > 0.0
    
    def test_identical_timelines(
        self,
        multi_event_timeline: EmotionTimeline
    ):
        dist = temporal_distance(
            multi_event_timeline,
            multi_event_timeline,
            window=timedelta(hours=1)
        )
        assert dist == 0.0
    
    def test_symmetric(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline
    ):
        dist_ab = temporal_distance(
            multi_event_timeline,
            sparse_timeline,
            window=timedelta(hours=1)
        )
        dist_ba = temporal_distance(
            sparse_timeline,
            multi_event_timeline,
            window=timedelta(hours=1)
        )
        assert dist_ab == dist_ba
    
    def test_deterministic(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline
    ):
        dist1 = temporal_distance(
            multi_event_timeline,
            sparse_timeline,
            window=timedelta(hours=1)
        )
        dist2 = temporal_distance(
            multi_event_timeline,
            sparse_timeline,
            window=timedelta(hours=1)
        )
        assert dist1 == dist2
    
    def test_non_overlapping_timelines(self, base_time: datetime):
        events_a = (
            EmotionEvent(timestamp=base_time, emotion_label="positive"),
        )
        events_b = (
            EmotionEvent(timestamp=base_time + timedelta(days=10), emotion_label="negative"),
        )
        timeline_a = EmotionTimeline(subject_id="a", events=events_a)
        timeline_b = EmotionTimeline(subject_id="b", events=events_b)
        
        dist = temporal_distance(timeline_a, timeline_b, window=timedelta(hours=1))
        assert dist > 0.0
    
    def test_perfectly_aligned_different_emotions(self, base_time: datetime):
        events_a = (
            EmotionEvent(timestamp=base_time, emotion_label="positive"),
            EmotionEvent(timestamp=base_time + timedelta(hours=1), emotion_label="positive"),
        )
        events_b = (
            EmotionEvent(timestamp=base_time + timedelta(minutes=30), emotion_label="negative"),
            EmotionEvent(timestamp=base_time + timedelta(hours=1, minutes=30), emotion_label="negative"),
        )
        timeline_a = EmotionTimeline(subject_id="a", events=events_a)
        timeline_b = EmotionTimeline(subject_id="b", events=events_b)
        
        dist = temporal_distance(timeline_a, timeline_b, window=timedelta(hours=1))
        assert dist == 0.0


# Tests for proximity_matrix

class TestProximityMatrix:
    
    def test_empty_input(self):
        result = proximity_matrix(timelines=(), window=timedelta(hours=1))
        assert result == []
    
    def test_single_timeline(self, single_event_timeline: EmotionTimeline):
        result = proximity_matrix(
            timelines=(single_event_timeline,),
            window=timedelta(hours=1)
        )
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] == 0.0
    
    def test_diagonal_is_zero(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline,
        dense_timeline: EmotionTimeline
    ):
        result = proximity_matrix(
            timelines=(multi_event_timeline, sparse_timeline, dense_timeline),
            window=timedelta(hours=1)
        )
        
        for i in range(len(result)):
            assert result[i][i] == 0.0
    
    def test_symmetric(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline,
        dense_timeline: EmotionTimeline
    ):
        result = proximity_matrix(
            timelines=(multi_event_timeline, sparse_timeline, dense_timeline),
            window=timedelta(hours=1)
        )
        
        n = len(result)
        for i in range(n):
            for j in range(n):
                assert result[i][j] == result[j][i]
    
    def test_square_matrix(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline
    ):
        result = proximity_matrix(
            timelines=(multi_event_timeline, sparse_timeline),
            window=timedelta(hours=1)
        )
        
        assert len(result) == 2
        assert all(len(row) == 2 for row in result)
    
    def test_deterministic(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline,
        dense_timeline: EmotionTimeline
    ):
        result1 = proximity_matrix(
            timelines=(multi_event_timeline, sparse_timeline, dense_timeline),
            window=timedelta(hours=1)
        )
        result2 = proximity_matrix(
            timelines=(multi_event_timeline, sparse_timeline, dense_timeline),
            window=timedelta(hours=1)
        )
        
        assert result1 == result2
    
    def test_uses_temporal_distance(
        self,
        multi_event_timeline: EmotionTimeline,
        sparse_timeline: EmotionTimeline
    ):
        matrix = proximity_matrix(
            timelines=(multi_event_timeline, sparse_timeline),
            window=timedelta(hours=1)
        )
        
        direct_dist = temporal_distance(
            multi_event_timeline,
            sparse_timeline,
            window=timedelta(hours=1)
        )
        
        assert matrix[0][1] == direct_dist
        assert matrix[1][0] == direct_dist
    
    def test_all_empty_timelines(self, empty_timeline: EmotionTimeline):
        result = proximity_matrix(
            timelines=(empty_timeline, empty_timeline),
            window=timedelta(hours=1)
        )
        
        assert result == [[0.0, 0.0], [0.0, 0.0]]
    
    def test_mixed_empty_and_non_empty(
        self,
        empty_timeline: EmotionTimeline,
        single_event_timeline: EmotionTimeline
    ):
        result = proximity_matrix(
            timelines=(empty_timeline, single_event_timeline),
            window=timedelta(hours=1)
        )
        
        assert result[0][0] == 0.0
        assert result[1][1] == 0.0
        assert result[0][1] > 0.0
        assert result[0][1] == result[1][0]


# Edge case tests

class TestEdgeCases:
    
    def test_very_small_window(self, base_time: datetime):
        events = (
            EmotionEvent(timestamp=base_time, emotion_label="positive"),
            EmotionEvent(timestamp=base_time + timedelta(seconds=1), emotion_label="negative"),
        )
        timeline = EmotionTimeline(subject_id="tiny", events=events)
        
        result = align_timelines_by_window(
            timelines=(timeline,),
            window=timedelta(milliseconds=100),
            anchor="start"
        )
        
        assert len(result) >= 1
    
    def test_very_large_window(self, multi_event_timeline: EmotionTimeline):
        result = align_timelines_by_window(
            timelines=(multi_event_timeline,),
            window=timedelta(days=365),
            anchor="start"
        )
        
        assert len(result) == 1
        assert result[0][0] is not None
    
    def test_unequal_length_timelines(self, base_time: datetime):
        events_short = (
            EmotionEvent(timestamp=base_time, emotion_label="positive"),
        )
        events_long = tuple(
            EmotionEvent(
                timestamp=base_time + timedelta(hours=i),
                emotion_label="neutral"
            )
            for i in range(10)
        )
        
        short = EmotionTimeline(subject_id="short", events=events_short)
        long = EmotionTimeline(subject_id="long", events=events_long)
        
        result = align_timelines_by_window(
            timelines=(short, long),
            window=timedelta(hours=1),
            anchor="start"
        )
        
        for window_idx, events in result.items():
            assert len(events) == 2
        
        dist = temporal_distance(short, long, window=timedelta(hours=1))
        assert dist > 0.0
    
    def test_same_timestamp_different_timelines(self, base_time: datetime):
        event = EmotionEvent(timestamp=base_time, emotion_label="positive")
        timeline_a = EmotionTimeline(subject_id="a", events=(event,))
        timeline_b = EmotionTimeline(subject_id="b", events=(event,))
        
        dist = temporal_distance(timeline_a, timeline_b, window=timedelta(hours=1))
        assert dist == 0.0
