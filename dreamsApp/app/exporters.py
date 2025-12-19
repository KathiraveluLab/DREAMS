# app/exporters.py

from typing import Dict, List, Any
from ..analytics.emotion_timeline import EmotionTimeline


def timeline_to_dict(timeline: EmotionTimeline) -> Dict[str, Any]:
    """
    Convert EmotionTimeline to a JSON-serializable dictionary.
    
    Preserves event ordering and converts timestamps to ISO 8601 strings.
    No data modification occurs.
    
    Args:
        timeline: EmotionTimeline to export
    
    Returns:
        Dict with keys: subject_id, events (list), metadata (optional)
    """
    events_list = []
    for event in timeline.events:
        event_dict = {
            'timestamp': event.timestamp.isoformat(),
            'emotion_label': event.emotion_label,
        }
        if event.score is not None:
            event_dict['score'] = event.score
        if event.source_id is not None:
            event_dict['source_id'] = event.source_id
        if event.metadata is not None:
            event_dict['metadata'] = event.metadata
        events_list.append(event_dict)
    
    result = {
        'subject_id': timeline.subject_id,
        'events': events_list,
    }
    
    if timeline.metadata is not None:
        result['metadata'] = timeline.metadata
    
    return result


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
    if len(timeline.events) == 0:
        return {
            'subject_id': timeline.subject_id,
            'event_count': 0,
            'time_span': None,
            'events': [],
            'metadata': timeline.metadata,
        }
    
    first_timestamp = timeline.events[0].timestamp
    last_timestamp = timeline.events[-1].timestamp
    time_span_seconds = (last_timestamp - first_timestamp).total_seconds()
    
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
