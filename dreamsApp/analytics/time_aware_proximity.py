# dreamsApp/analytics/time_aware_proximity.py

"""
Time-aware proximity and comparison for EmotionTimeline objects.
Purely structural - no ML, statistics, or interpretation.
"""

from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List, Literal

from .emotion_timeline import EmotionTimeline, EmotionEvent


def align_timelines_by_window(
    timelines: Tuple[EmotionTimeline, ...],
    window: timedelta,
    anchor: Literal["start", "end", "explicit"],
    anchor_time: Optional[datetime] = None
) -> Dict[int, Tuple[Optional[EmotionEvent], ...]]:
    """
    Align timelines into fixed windows. At most one event per timeline per window.
    Returns dict mapping window index to tuple of events (None if missing).
    """
    if window <= timedelta(0):
        raise ValueError("Window must be a positive timedelta")
    
    if anchor == "explicit" and anchor_time is None:
        raise ValueError("anchor_time required when anchor='explicit'")
    
    if not timelines:
        return {}
    
    all_empty = all(timeline.is_empty() for timeline in timelines)
    if all_empty:
        return {}
    
    all_timestamps: List[datetime] = []
    for timeline in timelines:
        for event in timeline.events:
            all_timestamps.append(event.timestamp)
    
    if not all_timestamps:
        return {}
    
    global_start = min(all_timestamps)
    global_end = max(all_timestamps)
    
    if anchor == "start":
        anchor_dt = global_start
    elif anchor == "end":
        anchor_dt = global_end
    else:
        anchor_dt = anchor_time
    
    if anchor == "end":
        num_windows_before = 0
        temp = anchor_dt
        while temp > global_start:
            temp -= window
            num_windows_before += 1
        
        window_start_base = anchor_dt - (num_windows_before * window)
    else:
        window_start_base = anchor_dt
    
    total_span = global_end - window_start_base
    num_windows = max(1, int(total_span / window) + 1)
    
    result: Dict[int, Tuple[Optional[EmotionEvent], ...]] = {}
    
    for window_idx in range(num_windows):
        window_start = window_start_base + (window_idx * window)
        window_end = window_start + window
        
        events_in_window: List[Optional[EmotionEvent]] = []
        
        for timeline in timelines:
            selected_event: Optional[EmotionEvent] = None
            
            # Select earliest event in window (deterministic)
            for event in timeline.events:
                if window_start <= event.timestamp < window_end:
                    selected_event = event
                    break
            
            events_in_window.append(selected_event)
        
        result[window_idx] = tuple(events_in_window)
    
    return result


def temporal_distance(
    a: EmotionTimeline,
    b: EmotionTimeline,
    window: timedelta,
    anchor: Literal["start", "end", "explicit"] = "start",
    anchor_time: Optional[datetime] = None
) -> float:
    """
    Distance between timelines based on presence/absence in windows.
    Symmetric, deterministic. Zero only for perfect alignment.
    """
    # Empty timelines are treated as perfectly aligned by definition
    if a.is_empty() and b.is_empty():
        return 0.0
    
    if a.is_empty() or b.is_empty():
        non_empty = a if not a.is_empty() else b
        return float(len(non_empty.events))
    
    aligned = align_timelines_by_window(
        timelines=(a, b),
        window=window,
        anchor=anchor,
        anchor_time=anchor_time
    )
    
    if not aligned:
        return 0.0
    
    mismatches = 0
    total_windows = len(aligned)
    
    for window_idx, events in aligned.items():
        event_a = events[0]
        event_b = events[1]
        
        a_present = event_a is not None
        b_present = event_b is not None
        
        if a_present != b_present:
            mismatches += 1
    
    if total_windows == 0:
        return 0.0
    
    return float(mismatches)


def proximity_matrix(
    timelines: Tuple[EmotionTimeline, ...],
    window: timedelta
) -> List[List[float]]:
    """
    Pairwise distance matrix. Square, symmetric, diagonal = 0.
    """
    n = len(timelines)
    
    if n == 0:
        return []
    
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = temporal_distance(timelines[i], timelines[j], window)
            matrix[i][j] = dist
            matrix[j][i] = dist
    
    return matrix
