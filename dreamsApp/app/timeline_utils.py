# app/timeline_utils.py

"""
Deprecated: Timeline utilities have been moved to EmotionTimeline class methods.

Use the following EmotionTimeline methods instead:
- timeline.time_gaps() instead of compute_time_gaps(timeline)
- len(timeline) instead of event_count(timeline)
- timeline.start_time() and timeline.end_time() for boundary access
- Chronological ordering is enforced automatically via __post_init__
"""
