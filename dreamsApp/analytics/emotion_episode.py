# dreamsApp/analytics/emotion_episode.py

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

from .emotion_timeline import EmotionEvent


__all__ = ['Episode']


@dataclass(frozen=True)
class Episode:
    start_time: datetime
    end_time: datetime
    events: Tuple[EmotionEvent, ...] = ()
    source_subject_id: Optional[str] = None
    
    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            object.__setattr__(self, 'events', tuple(self.events))
        
        if self.start_time > self.end_time:
            raise ValueError(
                f"start_time must be <= end_time: "
                f"{self.start_time} > {self.end_time}"
            )
        
        for i, event in enumerate(self.events):
            if event.timestamp < self.start_time:
                raise ValueError(
                    f"Event at index {i} has timestamp {event.timestamp} "
                    f"before episode start_time {self.start_time}"
                )
            if event.timestamp >= self.end_time:
                raise ValueError(
                    f"Event at index {i} has timestamp {event.timestamp} "
                    f"at or after episode end_time {self.end_time}"
                )
        
        for i in range(len(self.events) - 1):
            if self.events[i].timestamp > self.events[i + 1].timestamp:
                raise ValueError(
                    f"Events must be chronologically ordered. "
                    f"Event at index {i} ({self.events[i].timestamp}) "
                    f"occurs after event at index {i + 1} ({self.events[i + 1].timestamp})"
                )
    
    def __len__(self) -> int:
        return len(self.events)
    
    def is_empty(self) -> bool:
        return len(self.events) == 0
    
    def duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    def contains_timestamp(self, timestamp: datetime) -> bool:
        return self.start_time <= timestamp < self.end_time
    
    @property
    def episode_id(self) -> str:
        """Compute a deterministic ID based on episode content."""
        content = f"{self.start_time.isoformat()}:{self.end_time.isoformat()}"
        for event in self.events:
            content += f":{event.timestamp.isoformat()}:{event.emotion_label}"
            if event.score is not None:
                content += f":{event.score}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]
    
    def to_dict(self) -> Dict[str, Any]:
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
        
        result: Dict[str, Any] = {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'events': events_list,
        }
        
        if self.source_subject_id is not None:
            result['source_subject_id'] = self.source_subject_id
        
        return result
