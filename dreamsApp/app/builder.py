# app/builder.py

from datetime import datetime
from typing import List, Dict, Any, Optional
from ..analytics.emotion_timeline import EmotionEvent, EmotionTimeline


def build_emotion_timeline(
    subject_id: str,
    records: List[Dict[str, Any]],
    timeline_metadata: Optional[Dict[str, Any]] = None
) -> EmotionTimeline:
    """
    Construct an EmotionTimeline from validated records.
    
    Sorts records by timestamp and creates EmotionEvent objects.
    No filtering, aggregation, smoothing, or inference is performed.
    
    Args:
        subject_id: Identifier for the subject (person)
        records: List of dicts, each with keys: timestamp, emotion_label,
                 and optionally: score, source_id, metadata
        timeline_metadata: Optional metadata for the timeline itself
    
    Returns:
        EmotionTimeline: Immutable ordered collection of EmotionEvent objects
    """
    # Sort records by timestamp
    sorted_records = sorted(records, key=lambda r: r['timestamp'])
    
    # Build EmotionEvent objects
    events = []
    for record in sorted_records:
        event = EmotionEvent(
            timestamp=record['timestamp'],
            emotion_label=record['emotion_label'],
            score=record.get('score'),
            source_id=record.get('source_id'),
            metadata=record.get('metadata')
        )
        events.append(event)
    
    # Create and return immutable timeline
    return EmotionTimeline(
        subject_id=subject_id,
        events=tuple(events),
        metadata=timeline_metadata
    )
