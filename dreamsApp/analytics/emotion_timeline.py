# dreamsApp/analytics/emotion_timeline.py

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List


__all__ = ['EmotionEvent', 'EmotionTimeline']


@dataclass(frozen=True)
class EmotionEvent:
    timestamp: datetime
    emotion_label: str
    score: Optional[float] = None
    source_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EmotionTimeline:
    subject_id: str
    events: Tuple[EmotionEvent, ...] = ()
    
    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            object.__setattr__(self, 'events', tuple(self.events))
        
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
        if len(self.events) < 2:
            return 0.0
        return (self.events[-1].timestamp - self.events[0].timestamp).total_seconds()
    
    def temporal_bounds(self) -> Optional[Tuple[datetime, datetime]]:
        if self.is_empty():
            return None
        return (self.events[0].timestamp, self.events[-1].timestamp)
    
    def compute_gaps(self) -> Tuple[float, ...]:
        if len(self.events) < 2:
            return ()
        gaps: List[float] = []
        for i in range(len(self.events) - 1):
            gap = (self.events[i + 1].timestamp - self.events[i].timestamp).total_seconds()
            gaps.append(gap)
        return tuple(gaps)
    
    def fingerprint(self) -> str:
        event_count = len(self.events)
        total_duration = self.duration()
        gaps = self.compute_gaps()
        
        gap_hash = hashlib.sha256()
        for gap in gaps:
            gap_hash.update(f"{gap:.6f}".encode('utf-8'))
        gap_digest = gap_hash.hexdigest()[:16]
        
        fingerprint_str = f"{event_count}:{total_duration:.6f}:{gap_digest}"
        return hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()[:32]
