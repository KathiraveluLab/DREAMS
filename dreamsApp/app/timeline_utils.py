# app/timeline_utils.py

from datetime import timedelta
from typing import List, Tuple
from ..analytics.emotion_timeline import EmotionTimeline


def is_chronologically_ordered(timeline: EmotionTimeline) -> bool:
    """
    Check if all events in the timeline are in chronological order.
    
    Args:
        timeline: EmotionTimeline to check
    
    Returns:
        bool: True if events are strictly ordered by timestamp, False otherwise
    """
    if len(timeline.events) <= 1:
        return True
    
    for i in range(len(timeline.events) - 1):
        if timeline.events[i].timestamp > timeline.events[i + 1].timestamp:
            return False
    
    return True


def compute_time_gaps(timeline: EmotionTimeline) -> List[timedelta]:
    """
    Compute time deltas between adjacent events.
    
    Returns a list of timedelta objects representing the time elapsed between
    each pair of consecutive events. For a timeline with N events, returns N-1 gaps.
    
    Args:
        timeline: EmotionTimeline to analyze
    
    Returns:
        List[timedelta]: Time differences between consecutive events
    """
    if len(timeline.events) <= 1:
        return []
    
    gaps = []
    for i in range(len(timeline.events) - 1):
        gap = timeline.events[i + 1].timestamp - timeline.events[i].timestamp
        gaps.append(gap)
    
    return gaps


def event_timestamps(timeline: EmotionTimeline) -> Tuple[object, ...]:
    """
    Extract all timestamps from the timeline in order.
    
    Args:
        timeline: EmotionTimeline to extract from
    
    Returns:
        Tuple: All timestamps from events
    """
    return tuple(event.timestamp for event in timeline.events)


def event_count(timeline: EmotionTimeline) -> int:
    """
    Get the number of events in the timeline.
    
    Args:
        timeline: EmotionTimeline to count
    
    Returns:
        int: Number of events
    """
    return len(timeline.events)
