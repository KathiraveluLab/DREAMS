# dreamsApp/analytics/emotion_segmentation.py

"""
Temporal Segmentation Utilities (PR-4)

Provides purely structural utilities for slicing and aligning EmotionTimeline objects
in time. This module defines HOW timelines are segmented and aligned, but does NOT
perform comparison, proximity calculation, inference, or visualization.

WHAT THIS MODULE DOES:
- Segment timelines into fixed-duration windows
- Split timelines based on temporal gaps (session boundaries)
- Align multiple timelines to shared time boundaries
- Return segments as new EmotionTimeline objects with preserved event data

WHAT THIS MODULE DOES NOT DO:
- Perform ML, inference, clustering, or prediction
- Compute distances, similarities, or proximity scores
- Aggregate or map emotion labels to numeric values
- Interpret emotions semantically or psychologically
- Read/write databases or files
- Expose APIs or Flask routes
- Visualize or render data
- Mutate existing EmotionTimeline objects

RELATIONSHIP TO OTHER MODULES:
- PR-2 (emotion_timeline.py): Provides EmotionTimeline and EmotionEvent abstractions
- PR-3 (emotion_proximity.py): Uses windowing logic but performs aggregation and comparison
- PR-4 (this module): Provides structural segmentation WITHOUT aggregation or comparison

This module fills a structural gap between raw EmotionTimeline objects and
comparison/visualization layers. It enables future layers to operate on
time-aligned segments without reimplementing segmentation logic.

All operations are:
- Deterministic (same input → same output)
- Immutable (returns new EmotionTimeline objects)
- Reversible (no events dropped, no data loss)
- Structural (no interpretation or inference)
- Side-effect free (pure functions)

DESIGN PRINCIPLES:
- Never drop events (empty segments are explicit)
- Never modify original timelines
- Preserve event order and timestamps
- No numeric scoring or aggregation
- Outputs are EmotionTimeline segments, not numeric values

INVARIANTS:
- All functions assume EmotionTimeline.events is ordered by timestamp (ascending)
- Window boundaries follow [start, end) convention (start inclusive, end exclusive)
- Empty segments are represented as EmotionTimeline objects with zero events
- All segmentation operations preserve total event count across segments

Dependencies:
- EmotionTimeline and EmotionEvent from emotion_timeline.py (PR-2)
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
    Immutable representation of a time window with explicit boundaries.
    
    Used to define segment boundaries for timeline segmentation and alignment.
    
    Window boundaries follow the convention [start_time, end_time):
    - start_time is inclusive (events AT start_time are included)
    - end_time is exclusive (events AT end_time belong to the next window)
    
    Attributes:
        start_time: Start of the window (inclusive)
        end_time: End of the window (exclusive)
        index: Optional numeric index for ordered sequences of windows
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
        """
        Check if a timestamp falls within this window [start, end).
        
        Returns True if start_time <= timestamp < end_time.
        """
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
    Segment an EmotionTimeline into fixed-duration windows.
    
    Each segment is returned as a new EmotionTimeline containing only the events
    that fall within that window's time boundaries. The original timeline is
    never modified.
    
    This function performs ONLY structural slicing. It does NOT:
    - Aggregate emotion scores
    - Map emotion labels to numeric values
    - Interpolate missing data
    - Drop empty segments (empty segments are included with empty timelines)
    
    Window boundaries are defined as [start, end) - events at start are included,
    events at end belong to the next window.
    
    Design notes:
    - Windows span the full range from first to last event
    - Empty windows are included in output (with empty EmotionTimeline objects)
    - Window indices can be negative if events occur before anchor_time
    - All events are preserved across segments (no data loss)
    - Assumes timeline.events is ordered by timestamp (ascending)
    
    Window index calculation:
    - Uses floor division to assign events to windows
    - For events before anchor_time, this produces negative indices correctly
    - Example: event at anchor-30s with 60s windows → index -1 (not 0)
    
    Args:
        timeline: EmotionTimeline to segment
        window_duration: Duration of each fixed window (must be positive)
        anchor_time: Reference time for window alignment.
                    Defaults to timeline.start_time() if not provided.
                    Must be provided if timeline is empty.
    
    Returns:
        List of (TimeWindow, EmotionTimeline) tuples, ordered by window index.
        Each EmotionTimeline contains only events from that time window.
        Empty windows are included with empty EmotionTimeline objects.
    
    Raises:
        TypeError: If timeline is not an EmotionTimeline
        TypeError: If window_duration is not a timedelta
        ValueError: If window_duration is not positive
        ValueError: If timeline is empty and no anchor_time provided
    
    Example:
        >>> # Timeline with events at t=0s, t=30s, t=90s, t=120s
        >>> # Window duration = 60s
        >>> # Result: 3 windows
        >>> #   [0s-60s): events at 0s, 30s
        >>> #   [60s-120s): event at 90s
        >>> #   [120s-180s): event at 120s
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
    Split an EmotionTimeline at points where time gaps exceed a threshold.
    
    This is useful for identifying natural session boundaries or recording breaks.
    Each continuous sequence of events (where gaps < threshold) becomes a separate
    segment with its own EmotionTimeline.
    
    This function performs ONLY structural splitting based on temporal gaps.
    It does NOT:
    - Interpret the meaning of gaps (e.g., sleep, session breaks)
    - Perform clustering or pattern detection
    - Aggregate or analyze segment characteristics
    
    Design notes:
    - Segments are ordered chronologically
    - Each segment's TimeWindow spans from first to last event in that segment
    - No events are dropped
    - Single-event segments are valid
    - Empty timelines return empty list
    - Assumes timeline.events is ordered by timestamp (ascending)
    
    TimeWindow end_time convention:
    - For segments, end_time is set to last_event.timestamp + 1 microsecond
    - This ensures [start, end) semantics while keeping segments minimal
    - Microsecond precision chosen to avoid overlap with typical event spacing
    
    Args:
        timeline: EmotionTimeline to split
        gap_threshold: Minimum gap duration to trigger a split (must be positive)
    
    Returns:
        List of (TimeWindow, EmotionTimeline) tuples, ordered chronologically.
        Each segment is a continuous sequence where inter-event gaps < threshold.
        TimeWindow spans the segment's start to end time.
    
    Raises:
        TypeError: If timeline is not an EmotionTimeline
        TypeError: If gap_threshold is not a timedelta
        ValueError: If gap_threshold is not positive
    
    Example:
        >>> # Timeline with events at t=0s, t=10s, t=60s, t=65s
        >>> # Gap threshold = 30s
        >>> # Result: 2 segments
        >>> #   Segment 1: [0s-10s] with events at 0s, 10s
        >>> #   Segment 2: [60s-65s] with events at 60s, 65s
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
    Align multiple EmotionTimelines to shared window boundaries.
    
    For each provided window, extracts the subset of events from each timeline
    that fall within that window's boundaries. Returns aligned segments as new
    EmotionTimeline objects.
    
    This function performs ONLY structural alignment and slicing. It does NOT:
    - Compare or compute distances between aligned segments
    - Aggregate or score segments
    - Interpolate missing data
    - Drop timelines or windows with no events
    
    Design notes:
    - Each output EmotionTimeline contains only events within window boundaries
    - Empty timelines are included (for timelines with no events in a window)
    - Original timelines are never modified
    - Window indices from TimeWindow.index are used as dict keys
    - All events across all timelines are preserved
    - Assumes each timeline's events are ordered by timestamp (ascending)
    
    Args:
        timelines: List of EmotionTimeline objects to align
        windows: List of TimeWindow objects defining alignment boundaries
    
    Returns:
        Dict mapping window index to list of aligned EmotionTimeline segments.
        For window index i: dict[i] = [timeline_1_segment, timeline_2_segment, ...]
        Empty segments are included as empty EmotionTimeline objects.
    
    Raises:
        TypeError: If timelines is not a list
        TypeError: If windows is not a list
        TypeError: If any timeline is not an EmotionTimeline
        TypeError: If any window is not a TimeWindow
        ValueError: If timelines list is empty
        ValueError: If windows list is empty
    
    Example:
        >>> # Two timelines, three windows
        >>> # Timeline 1: events at 0s, 30s, 90s
        >>> # Timeline 2: events at 15s, 75s
        >>> # Windows: [0-60s), [60-120s), [120-180s)
        >>> # Result:
        >>> #   Window 0: [timeline1(0s,30s), timeline2(15s)]
        >>> #   Window 1: [timeline1(90s), timeline2(75s)]
        >>> #   Window 2: [timeline1(empty), timeline2(empty)]
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
            segment_timeline = EmotionTimeline(events_in_window)
            aligned_segments.append(segment_timeline)
        
        aligned[window_index] = aligned_segments
    
    return aligned
