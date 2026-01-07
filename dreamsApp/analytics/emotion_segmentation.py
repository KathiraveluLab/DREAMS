# dreamsApp/analytics/emotion_segmentation.py

"""
Temporal Segmentation Utilities (PR-4)

Provides structural utilities for slicing and aligning EmotionTimeline objects.
Performs segmentation and alignment without aggregation or comparison.

Key Features:
- Fixed-duration window segmentation
- Gap-based timeline splitting
- Multi-timeline alignment to shared windows

Design Principles:
- Immutable operations (returns new EmotionTimeline objects)
- Preserves all events (no data loss)
- Deterministic and side-effect free
- Window boundaries: [start, end) convention

Dependencies: EmotionTimeline and EmotionEvent from emotion_timeline.py
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .emotion_timeline import EmotionTimeline, EmotionEvent


__all__ = [
    'TimeWindow',
    'segment_timeline_fixed_windows',
    'segment_timeline_by_gaps',
    'align_timelines_to_windows',
]


@dataclass(frozen=True)
class TimeWindow:
    """
    Time window with explicit boundaries [start_time, end_time).
    
    Attributes:
        start_time: Start of window (inclusive)
        end_time: End of window (exclusive)
        index: Optional numeric index
    """
    start_time: datetime
    end_time: datetime
    index: Optional[int] = None
    
    def __post_init__(self):
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time must be after start_time: {self.start_time} >= {self.end_time}")
    
    def duration(self) -> timedelta:
        """Return the duration of this window."""
        return self.end_time - self.start_time
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within [start, end)."""
        return self.start_time <= timestamp < self.end_time
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        idx_str = f"idx={self.index}, " if self.index is not None else ""
        return f"TimeWindow({idx_str}{self.start_time.isoformat()} to {self.end_time.isoformat()})"


def segment_timeline_fixed_windows(
    timeline: EmotionTimeline,
    window_duration: timedelta,
    anchor_time: Optional[datetime] = None
) -> List[Tuple[TimeWindow, EmotionTimeline]]:
    """
    Segment timeline into fixed-duration windows.
    
    Returns new EmotionTimeline objects for each window. Preserves all events.
    Empty windows included. Window boundaries: [start, end).
    
    Args:
        timeline: EmotionTimeline to segment
        window_duration: Duration of each window (must be positive)
        anchor_time: Reference time for alignment (defaults to timeline start)
    
    Returns:
        List of (TimeWindow, EmotionTimeline) tuples ordered by window index.
    
    Raises:
        TypeError: If arguments have wrong types
        ValueError: If window_duration <= 0 or empty timeline without anchor_time
    """
    if not isinstance(timeline, EmotionTimeline):
        raise TypeError(f"timeline must be an EmotionTimeline, got {type(timeline).__name__}")
    if not isinstance(window_duration, timedelta):
        raise TypeError(f"window_duration must be a timedelta, got {type(window_duration).__name__}")
    if window_duration <= timedelta(0):
        raise ValueError("window_duration must be positive")
    
    # Handle empty timeline
    if timeline.is_empty():
        if anchor_time is None:
            raise ValueError("Cannot segment empty timeline without anchor_time")
        return []
    
    # Determine anchor time
    if anchor_time is None:
        anchor_time = timeline.start_time()
    
    # Compute window range
    first_timestamp = timeline.start_time()
    last_timestamp = timeline.end_time()
    
    # Calculate window indices for first and last events
    first_offset = (first_timestamp - anchor_time).total_seconds()
    last_offset = (last_timestamp - anchor_time).total_seconds()
    window_seconds = window_duration.total_seconds()
    
    first_window_idx = int(first_offset // window_seconds)
    last_window_idx = int(last_offset // window_seconds)
    
    # Generate all windows in range (including empty ones)
    segments: List[Tuple[TimeWindow, EmotionTimeline]] = []
    
    for window_idx in range(first_window_idx, last_window_idx + 1):
        # Define window boundaries
        window_start = anchor_time + timedelta(seconds=window_idx * window_seconds)
        window_end = anchor_time + timedelta(seconds=(window_idx + 1) * window_seconds)
        
        window = TimeWindow(
            start_time=window_start,
            end_time=window_end,
            index=window_idx
        )
        
        # Filter events that fall within this window
        events_in_window = [
            event for event in timeline.events
            if window.contains(event.timestamp)
        ]
        
        # Create new EmotionTimeline for this segment
        segment_timeline = EmotionTimeline(events_in_window)
        
        segments.append((window, segment_timeline))
    
    return segments


def segment_timeline_by_gaps(
    timeline: EmotionTimeline,
    gap_threshold: timedelta
) -> List[Tuple[TimeWindow, EmotionTimeline]]:
    """
    Split timeline at points where time gaps exceed threshold.
    
    Identifies session boundaries or recording breaks. Each continuous sequence
    becomes a separate segment.
    
    Args:
        timeline: EmotionTimeline to split
        gap_threshold: Minimum gap to trigger split (must be positive)
    
    Returns:
        List of (TimeWindow, EmotionTimeline) tuples ordered chronologically.
        Each segment contains continuous events with gaps < threshold.
    
    Raises:
        TypeError: If arguments have wrong types
        ValueError: If gap_threshold <= 0
    """
    if not isinstance(timeline, EmotionTimeline):
        raise TypeError(f"timeline must be an EmotionTimeline, got {type(timeline).__name__}")
    if not isinstance(gap_threshold, timedelta):
        raise TypeError(f"gap_threshold must be a timedelta, got {type(gap_threshold).__name__}")
    if gap_threshold <= timedelta(0):
        raise ValueError("gap_threshold must be positive")
    
    # Handle empty timeline
    if timeline.is_empty():
        return []
    
    # Handle single event
    if len(timeline.events) == 1:
        event = timeline.events[0]
        window = TimeWindow(
            start_time=event.timestamp,
            end_time=event.timestamp + timedelta(microseconds=1),
            index=0
        )
        segment_timeline = EmotionTimeline([event])
        return [(window, segment_timeline)]
    
    # Split based on gaps
    segments: List[Tuple[TimeWindow, EmotionTimeline]] = []
    current_segment_events = [timeline.events[0]]
    segment_start = timeline.events[0].timestamp
    
    for i in range(1, len(timeline.events)):
        prev_event = timeline.events[i - 1]
        curr_event = timeline.events[i]
        gap = curr_event.timestamp - prev_event.timestamp
        
        if gap >= gap_threshold:
            # Gap exceeds threshold - finalize current segment
            segment_end = prev_event.timestamp + timedelta(microseconds=1)
            window = TimeWindow(
                start_time=segment_start,
                end_time=segment_end,
                index=len(segments)
            )
            segment_timeline = EmotionTimeline(current_segment_events)
            segments.append((window, segment_timeline))
            
            # Start new segment
            current_segment_events = [curr_event]
            segment_start = curr_event.timestamp
        else:
            # Continue current segment
            current_segment_events.append(curr_event)
    
    # Finalize last segment
    segment_end = timeline.events[-1].timestamp + timedelta(microseconds=1)
    window = TimeWindow(
        start_time=segment_start,
        end_time=segment_end,
        index=len(segments)
    )
    segment_timeline = EmotionTimeline(current_segment_events)
    segments.append((window, segment_timeline))
    
    return segments


def align_timelines_to_windows(
    timelines: List[EmotionTimeline],
    windows: List[TimeWindow]
) -> Dict[int, List[EmotionTimeline]]:
    """
    Align multiple timelines to shared window boundaries.
    
    Extracts events from each timeline that fall within each window.
    Returns aligned segments as new EmotionTimeline objects.
    
    Args:
        timelines: List of EmotionTimeline objects to align
        windows: List of TimeWindow objects defining boundaries
    
    Returns:
        Dict mapping window index to list of aligned segments.
        Format: {window_index: [timeline_1_segment, timeline_2_segment, ...]}
        Empty segments included as empty EmotionTimeline objects.
    
    Raises:
        TypeError: If arguments have wrong types
        ValueError: If timelines or windows lists are empty
    """
    # Validate inputs
    if not isinstance(timelines, list):
        raise TypeError(f"timelines must be a list, got {type(timelines).__name__}")
    if not isinstance(windows, list):
        raise TypeError(f"windows must be a list, got {type(windows).__name__}")
    if not timelines:
        raise ValueError("timelines list cannot be empty")
    if not windows:
        raise ValueError("windows list cannot be empty")
    
    # Validate timeline types
    for i, timeline in enumerate(timelines):
        if not isinstance(timeline, EmotionTimeline):
            raise TypeError(f"timelines[{i}] must be an EmotionTimeline, got {type(timeline).__name__}")
    
    # Validate window types
    for i, window in enumerate(windows):
        if not isinstance(window, TimeWindow):
            raise TypeError(f"windows[{i}] must be a TimeWindow, got {type(window).__name__}")
    
    # Align each timeline to each window
    aligned: Dict[int, List[EmotionTimeline]] = {}
    
    for window in windows:
        window_index = window.index if window.index is not None else windows.index(window)
        aligned_segments = []
        
        for timeline in timelines:
            # Extract events within this window
            events_in_window = [
                event for event in timeline.events
                if window.contains(event.timestamp)
            ]
            
            # Create new EmotionTimeline for this segment
            segment_timeline = EmotionTimeline(subject_id=timeline.subject_id, events=tuple(events_in_window))
            aligned_segments.append(segment_timeline)
        
        aligned[window_index] = aligned_segments
    
    return aligned
