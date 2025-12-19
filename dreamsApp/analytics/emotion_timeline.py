# dreamsApp/analytics/emotion_timeline.py

"""
Time-Aware Emotion Timeline Engine

Provides immutable, chronologically-ordered structural containers for temporal
emotion data. This module is PURELY STRUCTURAL and does NOT perform:
- Sentiment analysis or inference
- Trend detection or prediction
- Statistical aggregation or smoothing
- Emotion interpretation or classification
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List


@dataclass(frozen=True)
class EmotionEvent:
    """
    A single emotional observation tied to a timestamp.
    
    Immutable structural container for one data point in an emotion timeline.
    All validation is assumed to occur upstream.
    
    Expected emotion_label values: 'positive', 'negative', 'neutral'
    (any string accepted; interpretation happens elsewhere)
    
    Attributes:
        timestamp: When the emotion was observed
        emotion_label: Emotion category (e.g., 'positive', 'negative', 'neutral')
        score: Optional intensity/confidence value
        source_id: Optional identifier for data origin (e.g., 'video_analysis')
        metadata: Optional additional context
    
    Does NOT:
        - Validate or normalize scores
        - Interpret or infer emotions
        - Validate timestamps
    """
    timestamp: datetime
    emotion_label: str
    score: Optional[float] = None
    source_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EmotionTimeline:
    """
    Chronologically-ordered, immutable collection of EmotionEvent objects.
    
    Single reusable temporal abstraction for emotion data across one subject.
    Enforces chronological ordering and provides lightweight temporal utilities.
    
    This is a structural container only—it does NOT perform trend analysis,
    volatility detection, proximity logic, or statistical operations.
    
    Attributes:
        subject_id: Identifier for the person/entity
        events: Immutable tuple of EmotionEvent objects (must be chronological)
        metadata: Optional timeline-level metadata
    
    Does NOT:
        - Analyze or interpret emotions
        - Detect trends or predict future states
        - Aggregate or smooth data
        - Filter or modify events
    """
    subject_id: str
    events: Tuple[EmotionEvent, ...] = ()
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Enforce chronological ordering of events.
        
        Raises:
            ValueError: If events are not in strict chronological order
        """
        if len(self.events) > 1:
            for i in range(len(self.events) - 1):
                if self.events[i].timestamp > self.events[i + 1].timestamp:
                    raise ValueError(
                        f"Events must be chronologically ordered. "
                        f"Event at index {i} ({self.events[i].timestamp}) "
                        f"occurs after event at index {i + 1} ({self.events[i + 1].timestamp})"
                    )
    
    def __len__(self) -> int:
        """Number of events in timeline."""
        return len(self.events)
    
    def is_empty(self) -> bool:
        """Check if timeline has no events."""
        return len(self.events) == 0
    
    def start_time(self) -> Optional[datetime]:
        """
        Timestamp of the first event.
        
        Returns:
            datetime of first event, or None if timeline is empty
        """
        return None if self.is_empty() else self.events[0].timestamp
    
    def end_time(self) -> Optional[datetime]:
        """
        Timestamp of the last event.
        
        Returns:
            datetime of last event, or None if timeline is empty
        """
        return None if self.is_empty() else self.events[-1].timestamp
    
    def time_span(self) -> Optional[timedelta]:
        """
        Total time span from first to last event.
        
        Returns:
            timedelta between first and last event, or None if timeline
            has fewer than 2 events
        """
        if len(self.events) < 2:
            return None
        return self.events[-1].timestamp - self.events[0].timestamp
    
    def time_gaps(self) -> Tuple[timedelta, ...]:
        """
        Time deltas between consecutive events.
        
        Returns tuple of timedelta objects representing gaps between adjacent
        events. For N events, returns N-1 gaps. Empty tuple if < 2 events.
        
        This is a lightweight structural helper—does NOT analyze or interpret gaps.
        """
        if len(self.events) < 2:
            return ()
        
        gaps = []
        for i in range(len(self.events) - 1):
            gaps.append(self.events[i + 1].timestamp - self.events[i].timestamp)
        return tuple(gaps)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export as JSON-serializable dictionary.
        
        Converts timestamps to ISO 8601 format. Does NOT modify, aggregate,
        or interpret data.
        
        Returns:
            Dict with keys: subject_id, events (list), metadata (optional)
        """
        events_list = []
        for event in self.events:
            event_dict = {
                'timestamp': event.timestamp.isoformat(),
                'emotion_label': event.emotion_label,
            }
            if event.score is not None:
                event_dict['score'] = event.score
            if event.source_id is not None:
                event_dict['source_id'] = event.source_id
            if event.metadata is not None:
                event_dict['metadata'] = event.metadata
            events_list.append(event_dict)
        
        result = {
            'subject_id': self.subject_id,
            'events': events_list,
        }
        
        if self.metadata is not None:
            result['metadata'] = self.metadata
        
        return result
    
    @classmethod
    def from_events(
        cls,
        subject_id: str,
        events: List[EmotionEvent],
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'EmotionTimeline':
        """
        Construct timeline from unsorted list of EmotionEvent objects.
        
        Convenience constructor that sorts events by timestamp before
        creating the immutable timeline.
        
        Args:
            subject_id: Identifier for the subject
            events: List of EmotionEvent objects (will be sorted by timestamp)
            metadata: Optional timeline-level metadata
        
        Returns:
            EmotionTimeline with chronologically ordered events
        """
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        return cls(
            subject_id=subject_id,
            events=tuple(sorted_events),
            metadata=metadata
        )
