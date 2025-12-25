# app/exporters.py

"""
Export utilities for EmotionTimeline.

Note: For basic JSON export, use timeline.to_dict() method directly.
This module provides additional specialized export formats.
"""

from typing import Dict, List, Any
from ..analytics.emotion_timeline import EmotionTimeline


def timeline_to_csv_rows(timeline: EmotionTimeline) -> List[Dict[str, Any]]:
    """
    Convert EmotionTimeline to a list of CSV-compatible dicts.
    
    Each dict represents one event row. Suitable for exporting to CSV
    or pandas DataFrames for analysis.
    
    Args:
        timeline: EmotionTimeline to export
    
    Returns:
        List of dicts, each representing an event with flattened fields
    """
    rows = []
    for event in timeline.events:
        row = {
            'subject_id': timeline.subject_id,
            'timestamp': event.timestamp.isoformat(),
            'emotion_label': event.emotion_label,
            'score': event.score,
            'source_id': event.source_id,
        }
        rows.append(row)
    
    return rows


def timeline_events_summary(timeline: EmotionTimeline) -> Dict[str, Any]:
    """
    Export timeline as a summary with event list and metadata.
    
    Suitable for research notebooks and dashboard visualization.
    Preserves all information in a structure-friendly format.
    
    Args:
        timeline: EmotionTimeline to summarize
    
    Returns:
        Dict with subject_id, event_count, time_span, and events
    """
    if timeline.is_empty():
        return {
            'subject_id': timeline.subject_id,
            'event_count': 0,
            'time_span_seconds': None,
            'first_event': None,
            'last_event': None,
            'events': [],
            'metadata': timeline.metadata,
        }
    
    first_timestamp = timeline.start_time()
    last_timestamp = timeline.end_time()
    time_span = timeline.time_span()
    time_span_seconds = time_span.total_seconds() if time_span is not None else 0.0
    
    events_data = []
    for i, event in enumerate(timeline.events):
        events_data.append({
            'index': i,
            'timestamp': event.timestamp.isoformat(),
            'emotion_label': event.emotion_label,
            'score': event.score,
            'source_id': event.source_id,
            'metadata': event.metadata,
        })
    
    return {
        'subject_id': timeline.subject_id,
        'event_count': len(timeline.events),
        'time_span_seconds': time_span_seconds,
        'first_event': first_timestamp.isoformat(),
        'last_event': last_timestamp.isoformat(),
        'events': events_data,
        'metadata': timeline.metadata,
    }
