# tests/test_timeline.py

import pytest
from datetime import datetime, timedelta
from dataclasses import FrozenInstanceError
from dreamsApp.analytics.emotion_timeline import EmotionEvent, EmotionTimeline
from dreamsApp.app.builder import build_emotion_timeline
from dreamsApp.app.timeline_utils import (
    is_chronologically_ordered,
    compute_time_gaps,
    event_count
)


class TestEmotionEventImmutability:
    """Test that EmotionEvent is immutable."""
    
    def test_emotion_event_frozen(self):
        """EmotionEvent should not allow attribute modification."""
        event = EmotionEvent(
            timestamp=datetime(2024, 12, 1, 8, 0, 0),
            emotion_label='joy',
            score=0.8
        )
        
        with pytest.raises(FrozenInstanceError):
            event.score = 0.5
    
    def test_emotion_event_creation(self):
        """EmotionEvent should be created with required fields."""
        event = EmotionEvent(
            timestamp=datetime(2024, 12, 1, 8, 0, 0),
            emotion_label='sadness'
        )
        assert event.emotion_label == 'sadness'
        assert event.score is None


class TestEmotionTimelineImmutability:
    """Test that EmotionTimeline is immutable."""
    
    def test_timeline_frozen(self):
        """EmotionTimeline should not allow attribute modification."""
        event = EmotionEvent(
            timestamp=datetime(2024, 12, 1, 8, 0, 0),
            emotion_label='joy'
        )
        timeline = EmotionTimeline(
            subject_id='person_01',
            events=(event,)
        )
        
        with pytest.raises(FrozenInstanceError):
            timeline.subject_id = 'person_02'
    
    def test_timeline_events_tuple(self):
        """EmotionTimeline events should be a tuple (immutable)."""
        event1 = EmotionEvent(
            timestamp=datetime(2024, 12, 1, 8, 0, 0),
            emotion_label='joy'
        )
        timeline = EmotionTimeline(
            subject_id='person_01',
            events=(event1,)
        )
        
        assert isinstance(timeline.events, tuple)
        with pytest.raises(TypeError):
            timeline.events[0] = None


class TestTimelineBuilder:
    """Test timeline builder sorts events by timestamp."""
    
    def test_builder_sorts_by_timestamp(self):
        """build_emotion_timeline should sort records by timestamp."""
        records = [
            {
                'timestamp': datetime(2024, 12, 1, 10, 0, 0),
                'emotion_label': 'joy',
                'score': 0.8
            },
            {
                'timestamp': datetime(2024, 12, 1, 8, 0, 0),
                'emotion_label': 'sadness',
                'score': 0.3
            },
            {
                'timestamp': datetime(2024, 12, 1, 9, 0, 0),
                'emotion_label': 'neutral',
                'score': 0.5
            },
        ]
        
        timeline = build_emotion_timeline('person_01', records)
        
        # Verify events are in chronological order
        assert timeline.events[0].timestamp == datetime(2024, 12, 1, 8, 0, 0)
        assert timeline.events[1].timestamp == datetime(2024, 12, 1, 9, 0, 0)
        assert timeline.events[2].timestamp == datetime(2024, 12, 1, 10, 0, 0)
    
    def test_builder_preserves_event_data(self):
        """build_emotion_timeline should preserve all event data."""
        records = [
            {
                'timestamp': datetime(2024, 12, 1, 8, 0, 0),
                'emotion_label': 'joy',
                'score': 0.6,
                'source_id': 'video_001'
            }
        ]
        
        timeline = build_emotion_timeline('person_01', records)
        event = timeline.events[0]
        
        assert event.emotion_label == 'joy'
        assert event.score == 0.6
        assert event.source_id == 'video_001'


class TestChronologicalOrdering:
    """Test is_chronologically_ordered utility."""
    
    def test_ordered_timeline(self):
        """is_chronologically_ordered should return True for ordered events."""
        events = (
            EmotionEvent(datetime(2024, 12, 1, 8, 0, 0), 'joy'),
            EmotionEvent(datetime(2024, 12, 1, 9, 0, 0), 'neutral'),
            EmotionEvent(datetime(2024, 12, 1, 10, 0, 0), 'sadness'),
        )
        timeline = EmotionTimeline('person_01', events)
        
        assert is_chronologically_ordered(timeline) is True
    
    def test_out_of_order_timeline(self):
        """is_chronologically_ordered should return False for unordered events."""
        events = (
            EmotionEvent(datetime(2024, 12, 1, 8, 0, 0), 'joy'),
            EmotionEvent(datetime(2024, 12, 1, 10, 0, 0), 'sadness'),
            EmotionEvent(datetime(2024, 12, 1, 9, 0, 0), 'neutral'),
        )
        timeline = EmotionTimeline('person_01', events)
        
        assert is_chronologically_ordered(timeline) is False
    
    def test_empty_timeline_ordered(self):
        """is_chronologically_ordered should return True for empty timeline."""
        timeline = EmotionTimeline('person_01', ())
        assert is_chronologically_ordered(timeline) is True
    
    def test_single_event_ordered(self):
        """is_chronologically_ordered should return True for single event."""
        events = (EmotionEvent(datetime(2024, 12, 1, 8, 0, 0), 'joy'),)
        timeline = EmotionTimeline('person_01', events)
        
        assert is_chronologically_ordered(timeline) is True


class TestTimeGaps:
    """Test compute_time_gaps utility."""
    
    def test_compute_gaps(self):
        """compute_time_gaps should return correct timedeltas."""
        events = (
            EmotionEvent(datetime(2024, 12, 1, 8, 0, 0), 'joy'),
            EmotionEvent(datetime(2024, 12, 1, 9, 30, 0), 'neutral'),
            EmotionEvent(datetime(2024, 12, 1, 11, 0, 0), 'sadness'),
        )
        timeline = EmotionTimeline('person_01', events)
        
        gaps = compute_time_gaps(timeline)
        
        assert len(gaps) == 2
        assert gaps[0] == timedelta(hours=1, minutes=30)
        assert gaps[1] == timedelta(hours=1, minutes=30)
    
    def test_gaps_empty_timeline(self):
        """compute_time_gaps should return empty list for empty timeline."""
        timeline = EmotionTimeline('person_01', ())
        gaps = compute_time_gaps(timeline)
        
        assert gaps == []
    
    def test_gaps_single_event(self):
        """compute_time_gaps should return empty list for single event."""
        events = (EmotionEvent(datetime(2024, 12, 1, 8, 0, 0), 'joy'),)
        timeline = EmotionTimeline('person_01', events)
        gaps = compute_time_gaps(timeline)
        
        assert gaps == []
