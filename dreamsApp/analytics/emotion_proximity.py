# dreamsApp/analytics/emotion_proximity.py

"""
Time-Aware Emotion Proximity Utility (PR-3)

Provides deterministic, structural comparison utilities for EmotionTimeline objects.
This module segments timelines into fixed time windows and computes simple numeric
distance metrics between aligned windows.

WHAT THIS MODULE DOES:
- Maps emotion labels to ordinal numeric values (positive=1, neutral=0, negative=-1)
- Segments an EmotionTimeline into fixed-duration time windows
- Aggregates emotion scores per window (mean of mapped values)
- Compares two timelines over aligned windows using simple distance metrics

WHAT THIS MODULE DOES NOT DO:
- Perform ML, inference, clustering, or learning
- Interpret emotions semantically or psychologically
- Detect trends, patterns, or anomalies
- Read from databases or external storage
- Persist results or expose APIs
- Visualize or render data
- Make assumptions about causality or meaning

All operations are:
- Deterministic (same input → same output, when anchor_time is provided)
- Reversible (no data loss in transformations)
- Structural (no interpretation or inference)
- Side-effect free (pure functions)

IMPORTANT DESIGN NOTES:
- When comparing two empty timelines without an explicit anchor_time,
  behavior is undefined; callers should always provide anchor_time for
  deterministic results in edge cases.
- Window indices can be negative if events occur before the anchor_time.
- Sparse window representation: only windows with events are stored.

Dependencies:
- EmotionTimeline and EmotionEvent from emotion_timeline.py (PR-2)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from .emotion_timeline import EmotionTimeline, EmotionEvent


__all__ = [
    'EMOTION_LABEL_MAP',
    'map_emotion_label',
    'segment_timeline_into_windows',
    'aggregate_window_scores',
    'get_aligned_window_range',
    'compare_timelines_distance',
    'compute_timeline_self_similarity',
]


# Simple ordinal mapping for emotion labels (6 basic emotions)
# This is a structural convention, NOT a semantic interpretation
# The numeric values are ordinal only; distances between them are not meaningful
# Using valence mapping: positive emotions > 0, negative emotions < 0
EMOTION_LABEL_MAP: Dict[str, float] = {
    'happiness': 1.0,
    'surprise': 0.5,
    'sadness': -0.5,
    'fear': -0.75,
    'anger': -0.8,
    'disgust': -1.0,
}


def map_emotion_label(label: str) -> float:
    """
    Map an emotion label to its numeric ordinal value.
    
    This mapping is purely structural for distance computation.
    It does NOT imply semantic meaning, intensity, or psychological interpretation.
    Uses 6 basic emotions: Happiness, Sadness, Fear, Anger, Disgust, Surprise.
    
    Args:
        label: Emotion label string (case-insensitive). Must be a non-empty string.
    
    Returns:
        Numeric value based on emotional valence:
        Happiness=1.0, Surprise=0.5, Sadness=-0.5, Fear=-0.75, Anger=-0.8, Disgust=-1.0
        Returns 0.0 for unrecognized labels
    
    Raises:
        TypeError: If label is not a string
        ValueError: If label is an empty string
    
    Example:
        >>> map_emotion_label('Happiness')
        1.0
        >>> map_emotion_label('ANGER')
        -0.8
        >>> map_emotion_label('unknown')
        0.0
    """
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}")
    if not label:
        raise ValueError("label must not be an empty string")
    return EMOTION_LABEL_MAP.get(label.lower(), 0.0)


def segment_timeline_into_windows(
    timeline: EmotionTimeline,
    window_duration: timedelta,
    anchor_time: Optional[datetime] = None
) -> Dict[int, List[EmotionEvent]]:
    """
    Segment an EmotionTimeline into fixed-duration time windows.
    
    Events are assigned to windows based on their timestamp. Window indices
    are integers starting from 0 at the anchor time. Each window spans
    [anchor + i*duration, anchor + (i+1)*duration).
    
    This is a structural grouping operation. It does NOT:
    - Interpolate missing windows
    - Smooth or aggregate data
    - Interpret temporal patterns
    
    Design notes:
    - Window indices can be negative if events precede anchor_time
    - Only windows containing events are returned (sparse representation)
    - Events exactly on window boundaries belong to the new window
    
    Args:
        timeline: EmotionTimeline to segment (must be an EmotionTimeline instance)
        window_duration: Duration of each window (must be positive timedelta)
        anchor_time: Reference time for window alignment.
                    Defaults to timeline.start_time() if not provided.
    
    Returns:
        Dict mapping window index (int) to list of EmotionEvent objects in that window.
        Only windows containing events are included (sparse representation).
    
    Raises:
        TypeError: If timeline is not an EmotionTimeline
        TypeError: If window_duration is not a timedelta
        ValueError: If window_duration is not positive
        ValueError: If timeline is empty and no anchor_time provided
    
    Example:
        >>> # Timeline with events at t=0s, t=30s, t=90s
        >>> # Window duration = 60s
        >>> # Result: {0: [event_0s, event_30s], 1: [event_90s]}
    """
    if not isinstance(timeline, EmotionTimeline):
        raise TypeError(f"timeline must be an EmotionTimeline, got {type(timeline).__name__}")
    if not isinstance(window_duration, timedelta):
        raise TypeError(f"window_duration must be a timedelta, got {type(window_duration).__name__}")
    if window_duration <= timedelta(0):
        raise ValueError("window_duration must be positive")
    
    if timeline.is_empty():
        if anchor_time is None:
            raise ValueError("Cannot segment empty timeline without anchor_time")
        return {}
    
    if anchor_time is None:
        anchor_time = timeline.start_time()
    
    windows: Dict[int, List[EmotionEvent]] = {}
    
    for event in timeline.events:
        # Compute window index for this event
        time_offset = event.timestamp - anchor_time
        offset_seconds = time_offset.total_seconds()
        window_seconds = window_duration.total_seconds()
        
        # Handle events before anchor time (negative indices)
        # Floor division ensures correct window assignment for negative offsets
        window_index = int(offset_seconds // window_seconds)
        
        if window_index not in windows:
            windows[window_index] = []
        windows[window_index].append(event)
    
    return windows


def aggregate_window_scores(
    windowed_events: Dict[int, List[EmotionEvent]],
    use_event_scores: bool = False
) -> Dict[int, float]:
    """
    Compute aggregate score for each window.
    
    For each window, computes the mean of either:
    - Mapped emotion labels (default): map_emotion_label(event.emotion_label)
    - Event scores (if use_event_scores=True and scores exist)
    
    This is a simple arithmetic mean with no weighting, smoothing, or
    statistical adjustment. It does NOT interpret or infer meaning.
    
    Design notes:
    - Windows with no valid scores (e.g., all None when use_event_scores=True)
      are omitted from output
    - Empty input returns empty dict
    
    Args:
        windowed_events: Dict from segment_timeline_into_windows()
        use_event_scores: If True, use event.score values instead of mapped labels.
                         Events without scores are skipped.
    
    Returns:
        Dict mapping window index to aggregate score (float).
        Windows with no valid scores are omitted.
    
    Raises:
        TypeError: If windowed_events is not a dict
    
    Example:
        >>> # Window 0 has ['Happiness', 'Surprise'] → mean([1.0, 0.5]) = 0.75
        >>> # Window 1 has ['Anger'] → mean([-0.8]) = -0.8
        >>> # Result: {0: 0.75, 1: -0.8}
    """
    if not isinstance(windowed_events, dict):
        raise TypeError(f"windowed_events must be a dict, got {type(windowed_events).__name__}")
    
    aggregates: Dict[int, float] = {}
    
    for window_index, events in windowed_events.items():
        if not events:
            # Skip empty event lists (should not occur from segment_timeline_into_windows,
            # but guard defensively)
            continue
        
        if use_event_scores:
            scores = [e.score for e in events if e.score is not None]
        else:
            scores = [map_emotion_label(e.emotion_label) for e in events]
        
        if scores:
            aggregates[window_index] = sum(scores) / len(scores)
    
    return aggregates


def get_aligned_window_range(
    scores_a: Dict[int, float],
    scores_b: Dict[int, float]
) -> Tuple[int, int]:
    """
    Determine the overlapping window index range between two score dicts.
    
    Returns the inclusive range [min_index, max_index] covering all windows
    present in either input. This defines the alignment range for comparison.
    
    Design notes:
    - Returns union of indices, not intersection
    - (0, -1) is a sentinel indicating empty/invalid range (max < min)
    
    Args:
        scores_a: Aggregated scores for timeline A
        scores_b: Aggregated scores for timeline B
    
    Returns:
        Tuple (min_index, max_index) of the union of window indices.
        Returns (0, -1) if both inputs are empty (indicating no valid range).
    
    Raises:
        TypeError: If either argument is not a dict
    
    Example:
        >>> get_aligned_window_range({0: 0.5, 2: -0.5}, {1: 0.0, 3: 1.0})
        (0, 3)
    """
    if not isinstance(scores_a, dict):
        raise TypeError(f"scores_a must be a dict, got {type(scores_a).__name__}")
    if not isinstance(scores_b, dict):
        raise TypeError(f"scores_b must be a dict, got {type(scores_b).__name__}")
    
    all_indices = set(scores_a.keys()) | set(scores_b.keys())
    
    if not all_indices:
        # Sentinel value: max < min indicates empty range
        return (0, -1)
    
    return (min(all_indices), max(all_indices))


def compare_timelines_distance(
    timeline_a: EmotionTimeline,
    timeline_b: EmotionTimeline,
    window_duration: timedelta,
    anchor_time: Optional[datetime] = None,
    use_event_scores: bool = False,
    missing_value: float = 0.0
) -> Dict[str, Any]:
    """
    Compare two EmotionTimelines using simple distance metrics over aligned windows.
    
    Segments both timelines into fixed windows, aggregates scores per window,
    then computes distance metrics across the aligned window range.
    
    Distance metrics computed:
    - mean_absolute_difference: Mean of |score_a - score_b| per window
    - sum_squared_difference: Sum of (score_a - score_b)^2 per window
    - window_count: Number of windows in aligned range
    - matched_windows: Number of windows with data in both timelines
    - per_window_differences: Dict of window_index → (score_a, score_b, difference)
    
    This is a structural comparison. It does NOT:
    - Interpret differences as meaningful changes
    - Detect trends or anomalies
    - Apply statistical tests or significance measures
    - Handle causality or directionality
    
    Design notes:
    - If both timelines are empty and no anchor_time is provided, the function
      uses datetime.now() as a fallback. For deterministic behavior, always
      provide an explicit anchor_time.
    - missing_value is used for windows that exist in the aligned range but
      have no events in one timeline (sparse window handling)
    
    Args:
        timeline_a: First EmotionTimeline
        timeline_b: Second EmotionTimeline
        window_duration: Duration of each window
        anchor_time: Reference time for alignment.
                    Defaults to earliest start_time of either timeline.
                    WARNING: If both timelines are empty, defaults to datetime.now()
                    which breaks determinism. Provide explicit anchor_time for
                    deterministic results.
        use_event_scores: Use event.score instead of mapped labels
        missing_value: Value to use when a window has no data (default 0.0)
    
    Returns:
        Dict containing:
        - mean_absolute_difference: float (or None if no windows)
        - sum_squared_difference: float
        - window_count: int
        - matched_windows: int
        - per_window_differences: Dict[int, Tuple[float, float, float]]
        - anchor_time: ISO 8601 string of datetime used for alignment
        - window_duration_seconds: float
    
    Raises:
        TypeError: If timeline_a or timeline_b is not an EmotionTimeline
        TypeError: If window_duration is not a timedelta
        ValueError: If window_duration is not positive
    
    Example:
        >>> result = compare_timelines_distance(timeline_a, timeline_b, timedelta(minutes=5))
        >>> result['mean_absolute_difference']
        0.25
    """
    if not isinstance(timeline_a, EmotionTimeline):
        raise TypeError(f"timeline_a must be an EmotionTimeline, got {type(timeline_a).__name__}")
    if not isinstance(timeline_b, EmotionTimeline):
        raise TypeError(f"timeline_b must be an EmotionTimeline, got {type(timeline_b).__name__}")
    if not isinstance(window_duration, timedelta):
        raise TypeError(f"window_duration must be a timedelta, got {type(window_duration).__name__}")
    if window_duration <= timedelta(0):
        raise ValueError("window_duration must be positive")
    
    # Determine anchor time for alignment
    if anchor_time is None:
        start_a = timeline_a.start_time()
        start_b = timeline_b.start_time()
        
        if start_a is None and start_b is None:
            # Both timelines are empty and no anchor_time was provided.
            # Raise an error to enforce deterministic behavior.
            raise ValueError("Cannot compare two empty timelines without an explicit anchor_time.")
        elif start_a is None:
            anchor_time = start_b
        elif start_b is None:
            anchor_time = start_a
        else:
            anchor_time = min(start_a, start_b)
    
    # Segment both timelines (anchor_time is guaranteed non-None here)
    windows_a = segment_timeline_into_windows(timeline_a, window_duration, anchor_time)
    windows_b = segment_timeline_into_windows(timeline_b, window_duration, anchor_time)
    
    # Aggregate scores per window
    scores_a = aggregate_window_scores(windows_a, use_event_scores)
    scores_b = aggregate_window_scores(windows_b, use_event_scores)
    
    # Get aligned range
    min_idx, max_idx = get_aligned_window_range(scores_a, scores_b)
    
    # Handle empty case (sentinel: max < min)
    if max_idx < min_idx:
        return {
            'mean_absolute_difference': None,
            'sum_squared_difference': 0.0,
            'window_count': 0,
            'matched_windows': 0,
            'per_window_differences': {},
            'anchor_time': anchor_time.isoformat(),
            'window_duration_seconds': window_duration.total_seconds(),
        }
    
    # Compute differences across aligned windows
    per_window_differences: Dict[int, Tuple[float, float, float]] = {}
    absolute_differences: List[float] = []
    sum_squared = 0.0
    matched_count = 0
    
    for idx in range(min_idx, max_idx + 1):
        score_a = scores_a.get(idx, missing_value)
        score_b = scores_b.get(idx, missing_value)
        diff = score_a - score_b
        
        per_window_differences[idx] = (score_a, score_b, diff)
        absolute_differences.append(abs(diff))
        sum_squared += diff * diff
        
        # Count as matched if both have actual data (not using missing_value)
        if idx in scores_a and idx in scores_b:
            matched_count += 1
    
    window_count = max_idx - min_idx + 1
    mean_abs_diff = sum(absolute_differences) / len(absolute_differences) if absolute_differences else None
    
    return {
        'mean_absolute_difference': mean_abs_diff,
        'sum_squared_difference': sum_squared,
        'window_count': window_count,
        'matched_windows': matched_count,
        'per_window_differences': per_window_differences,
        'anchor_time': anchor_time.isoformat(),
        'window_duration_seconds': window_duration.total_seconds(),
    }


def compute_timeline_self_similarity(
    timeline: EmotionTimeline,
    window_duration: timedelta,
    use_event_scores: bool = False
) -> Dict[str, Any]:
    """
    Compute self-similarity structure of a single timeline.
    
    Segments the timeline into windows and returns the score distribution
    across windows. This provides a structural fingerprint of the timeline
    without interpreting patterns or trends.
    
    Design notes:
    - Uses population variance (divides by N), not sample variance (N-1)
    - Empty timelines return zeroed structure
    - score_range is (0.0, 0.0) for empty or scoreless timelines
    
    Args:
        timeline: EmotionTimeline to analyze
        window_duration: Duration of each window (must be positive)
        use_event_scores: Use event.score instead of mapped labels
    
    Returns:
        Dict containing:
        - window_scores: Dict[int, float] of window index to score
        - score_range: Tuple[float, float] of (min_score, max_score)
        - score_variance: float (population variance of scores)
        - window_count: int
        - total_events: int
    
    Raises:
        TypeError: If timeline is not an EmotionTimeline
        TypeError: If window_duration is not a timedelta
        ValueError: If window_duration is not positive
    
    Does NOT interpret variance as volatility or detect meaningful patterns.
    """
    if not isinstance(timeline, EmotionTimeline):
        raise TypeError(f"timeline must be an EmotionTimeline, got {type(timeline).__name__}")
    if not isinstance(window_duration, timedelta):
        raise TypeError(f"window_duration must be a timedelta, got {type(window_duration).__name__}")
    if window_duration <= timedelta(0):
        raise ValueError("window_duration must be positive")
    
    if timeline.is_empty():
        return {
            'window_scores': {},
            'score_range': (0.0, 0.0),
            'score_variance': 0.0,
            'window_count': 0,
            'total_events': 0,
        }
    
    windows = segment_timeline_into_windows(timeline, window_duration)
    scores = aggregate_window_scores(windows, use_event_scores)
    
    if not scores:
        # All events had no valid scores (e.g., use_event_scores=True but all scores are None)
        return {
            'window_scores': {},
            'score_range': (0.0, 0.0),
            'score_variance': 0.0,
            'window_count': 0,
            'total_events': len(timeline),
        }
    
    score_values = list(scores.values())
    min_score = min(score_values)
    max_score = max(score_values)
    
    # Population variance (not sample variance) - divides by N, not N-1
    mean_score = sum(score_values) / len(score_values)
    variance = sum((s - mean_score) ** 2 for s in score_values) / len(score_values)
    
    return {
        'window_scores': scores,
        'score_range': (min_score, max_score),
        'score_variance': variance,
        'window_count': len(scores),
        'total_events': len(timeline),
    }
