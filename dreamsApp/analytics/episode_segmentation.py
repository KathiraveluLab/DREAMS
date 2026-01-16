# dreamsApp/analytics/episode_segmentation.py

from datetime import timedelta
from typing import List

from .emotion_timeline import EmotionTimeline
from .emotion_episode import Episode
from .emotion_segmentation import segment_timeline_by_gaps


__all__ = ['segment_timeline_to_episodes']


def segment_timeline_to_episodes(
    timeline: EmotionTimeline,
    gap_threshold: timedelta
) -> List[Episode]:
    if not isinstance(timeline, EmotionTimeline):
        raise TypeError(f"timeline must be an EmotionTimeline, got {type(timeline).__name__}")
    if not isinstance(gap_threshold, timedelta):
        raise TypeError(f"gap_threshold must be a timedelta, got {type(gap_threshold).__name__}")
    if gap_threshold <= timedelta(0):
        raise ValueError("gap_threshold must be positive")
    
    if timeline.is_empty():
        return []
    
    segments = segment_timeline_by_gaps(timeline, gap_threshold)
    
    episodes: List[Episode] = []
    
    for window, segment_timeline in segments:
        episode = Episode(
            start_time=window.start_time,
            end_time=window.end_time,
            events=segment_timeline.events,
            source_subject_id=timeline.subject_id
        )
        episodes.append(episode)
    
    return episodes
