# dreamsApp/analytics/emotion_segmentation.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

from .emotion_timeline import EmotionTimeline, EmotionEvent


__all__ = ['TimeWindow', 'segment_timeline_by_gaps']


@dataclass(frozen=True)
class TimeWindow:
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self) -> None:
        if self.start_time > self.end_time:
            raise ValueError(
                f"start_time must be <= end_time: "
                f"{self.start_time} > {self.end_time}"
            )
    
    def duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


def segment_timeline_by_gaps(
    timeline: EmotionTimeline,
    gap_threshold: timedelta
) -> List[Tuple[TimeWindow, EmotionTimeline]]:
    if not isinstance(timeline, EmotionTimeline):
        raise TypeError(f"timeline must be an EmotionTimeline, got {type(timeline).__name__}")
    if not isinstance(gap_threshold, timedelta):
        raise TypeError(f"gap_threshold must be a timedelta, got {type(gap_threshold).__name__}")
    if gap_threshold <= timedelta(0):
        raise ValueError("gap_threshold must be positive")
    
    if timeline.is_empty():
        return []
    
    segments: List[Tuple[TimeWindow, EmotionTimeline]] = []
    current_events: List[EmotionEvent] = [timeline.events[0]]
    
    for i in range(1, len(timeline.events)):
        prev_event = timeline.events[i - 1]
        curr_event = timeline.events[i]
        gap = curr_event.timestamp - prev_event.timestamp
        
        if gap > gap_threshold:
            window = TimeWindow(
                start_time=current_events[0].timestamp,
                end_time=current_events[-1].timestamp + timedelta(seconds=1)
            )
            segment_timeline = EmotionTimeline(
                subject_id=timeline.subject_id,
                events=tuple(current_events)
            )
            segments.append((window, segment_timeline))
            current_events = [curr_event]
        else:
            current_events.append(curr_event)
    
    if current_events:
        window = TimeWindow(
            start_time=current_events[0].timestamp,
            end_time=current_events[-1].timestamp + timedelta(seconds=1)
        )
        segment_timeline = EmotionTimeline(
            subject_id=timeline.subject_id,
            events=tuple(current_events)
        )
        segments.append((window, segment_timeline))
    
    return segments
