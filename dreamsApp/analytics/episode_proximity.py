# dreamsApp/analytics/episode_proximity.py

from datetime import datetime, timedelta
from enum import Enum
from typing import Tuple

from .emotion_episode import Episode


__all__ = [
    'ProximityRelation',
    'compute_temporal_overlap',
    'are_episodes_adjacent',
    'classify_episode_proximity',
]


class ProximityRelation(Enum):
    OVERLAPPING = "overlapping"
    ADJACENT = "adjacent"
    DISJOINT = "disjoint"


def compute_temporal_overlap(
    episode_a: Episode,
    episode_b: Episode
) -> float:
    if not isinstance(episode_a, Episode):
        raise TypeError(f"episode_a must be an Episode, got {type(episode_a).__name__}")
    if not isinstance(episode_b, Episode):
        raise TypeError(f"episode_b must be an Episode, got {type(episode_b).__name__}")
    
    overlap_start = max(episode_a.start_time, episode_b.start_time)
    overlap_end = min(episode_a.end_time, episode_b.end_time)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_duration = (overlap_end - overlap_start).total_seconds()
    
    duration_a = episode_a.duration()
    duration_b = episode_b.duration()
    
    if duration_a == 0.0 and duration_b == 0.0:
        return 1.0 if episode_a.start_time == episode_b.start_time else 0.0
    
    min_duration = min(duration_a, duration_b) if min(duration_a, duration_b) > 0 else max(duration_a, duration_b)
    
    return overlap_duration / min_duration


def compute_temporal_gap(
    episode_a: Episode,
    episode_b: Episode
) -> float:
    if not isinstance(episode_a, Episode):
        raise TypeError(f"episode_a must be an Episode, got {type(episode_a).__name__}")
    if not isinstance(episode_b, Episode):
        raise TypeError(f"episode_b must be an Episode, got {type(episode_b).__name__}")
    
    if episode_a.end_time <= episode_b.start_time:
        gap = (episode_b.start_time - episode_a.end_time).total_seconds()
    elif episode_b.end_time <= episode_a.start_time:
        gap = (episode_a.start_time - episode_b.end_time).total_seconds()
    else:
        gap = 0.0
    
    return max(0.0, gap)


def are_episodes_adjacent(
    episode_a: Episode,
    episode_b: Episode,
    adjacency_threshold: timedelta = timedelta(0)
) -> bool:
    if not isinstance(episode_a, Episode):
        raise TypeError(f"episode_a must be an Episode, got {type(episode_a).__name__}")
    if not isinstance(episode_b, Episode):
        raise TypeError(f"episode_b must be an Episode, got {type(episode_b).__name__}")
    if not isinstance(adjacency_threshold, timedelta):
        raise TypeError(f"adjacency_threshold must be a timedelta, got {type(adjacency_threshold).__name__}")
    if adjacency_threshold < timedelta(0):
        raise ValueError("adjacency_threshold must be non-negative")
    
    overlap = compute_temporal_overlap(episode_a, episode_b)
    if overlap > 0.0:
        return False
    
    gap_seconds = compute_temporal_gap(episode_a, episode_b)
    threshold_seconds = adjacency_threshold.total_seconds()
    
    return gap_seconds <= threshold_seconds


def classify_episode_proximity(
    episode_a: Episode,
    episode_b: Episode,
    adjacency_threshold: timedelta = timedelta(0)
) -> ProximityRelation:
    if not isinstance(episode_a, Episode):
        raise TypeError(f"episode_a must be an Episode, got {type(episode_a).__name__}")
    if not isinstance(episode_b, Episode):
        raise TypeError(f"episode_b must be an Episode, got {type(episode_b).__name__}")
    if not isinstance(adjacency_threshold, timedelta):
        raise TypeError(f"adjacency_threshold must be a timedelta, got {type(adjacency_threshold).__name__}")
    if adjacency_threshold < timedelta(0):
        raise ValueError("adjacency_threshold must be non-negative")
    
    overlap = compute_temporal_overlap(episode_a, episode_b)
    if overlap > 0.0:
        return ProximityRelation.OVERLAPPING
    
    gap_seconds = compute_temporal_gap(episode_a, episode_b)
    if gap_seconds <= adjacency_threshold.total_seconds():
        return ProximityRelation.ADJACENT
    
    return ProximityRelation.DISJOINT
