# dreamsApp/analytics/serialization.py

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

from .emotion_timeline import EmotionEvent, EmotionTimeline
from .emotion_episode import Episode
from .episode_proximity import ProximityRelation
from .temporal_narrative_graph import NarrativeEdge, TemporalNarrativeGraph


__all__ = [
    'SCHEMA_VERSION',
    'SerializedPayload',
    'EmotionTimelineSerializer',
    'EpisodeSerializer',
    'TemporalNarrativeGraphSerializer',
]


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SerializedPayload:
    data: Dict[str, Any]
    schema_version: str
    fingerprint: str
    
    def to_json(self) -> str:
        return json.dumps({
            'schema_version': self.schema_version,
            'fingerprint': self.fingerprint,
            'data': self.data,
        }, sort_keys=True, separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SerializedPayload':
        parsed = json.loads(json_str)
        return cls(
            data=parsed['data'],
            schema_version=parsed['schema_version'],
            fingerprint=parsed['fingerprint'],
        )


class EmotionTimelineSerializer:
    
    @staticmethod
    def serialize(timeline: EmotionTimeline) -> SerializedPayload:
        if not isinstance(timeline, EmotionTimeline):
            raise TypeError(f"Expected EmotionTimeline, got {type(timeline).__name__}")
        
        events_data: List[Dict[str, Any]] = []
        for event in timeline.events:
            event_dict: Dict[str, Any] = {
                'timestamp': event.timestamp.isoformat(),
                'emotion_label': event.emotion_label,
            }
            if event.score is not None:
                event_dict['score'] = event.score
            if event.source_id is not None:
                event_dict['source_id'] = event.source_id
            if event.metadata is not None:
                event_dict['metadata'] = event.metadata
            events_data.append(event_dict)
        
        data = {
            'subject_id': timeline.subject_id,
            'events': events_data,
        }
        
        return SerializedPayload(
            data=data,
            schema_version=SCHEMA_VERSION,
            fingerprint=timeline.fingerprint(),
        )
    
    @staticmethod
    def deserialize(payload: SerializedPayload) -> EmotionTimeline:
        if not isinstance(payload, SerializedPayload):
            raise TypeError(f"Expected SerializedPayload, got {type(payload).__name__}")
        
        data = payload.data
        events: List[EmotionEvent] = []
        
        for event_dict in data['events']:
            events.append(EmotionEvent(
                timestamp=datetime.fromisoformat(event_dict['timestamp']),
                emotion_label=event_dict['emotion_label'],
                score=event_dict.get('score'),
                source_id=event_dict.get('source_id'),
                metadata=event_dict.get('metadata'),
            ))
        
        return EmotionTimeline(
            subject_id=data['subject_id'],
            events=tuple(events),
        )


class EpisodeSerializer:
    
    @staticmethod
    def serialize(episode: Episode) -> SerializedPayload:
        if not isinstance(episode, Episode):
            raise TypeError(f"Expected Episode, got {type(episode).__name__}")
        
        events_data: List[Dict[str, Any]] = []
        for event in episode.events:
            event_dict: Dict[str, Any] = {
                'timestamp': event.timestamp.isoformat(),
                'emotion_label': event.emotion_label,
            }
            if event.score is not None:
                event_dict['score'] = event.score
            if event.source_id is not None:
                event_dict['source_id'] = event.source_id
            if event.metadata is not None:
                event_dict['metadata'] = event.metadata
            events_data.append(event_dict)
        
        data: Dict[str, Any] = {
            'start_time': episode.start_time.isoformat(),
            'end_time': episode.end_time.isoformat(),
            'events': events_data,
        }
        if episode.source_subject_id is not None:
            data['source_subject_id'] = episode.source_subject_id
        
        return SerializedPayload(
            data=data,
            schema_version=SCHEMA_VERSION,
            fingerprint=episode.episode_id,
        )
    
    @staticmethod
    def deserialize(payload: SerializedPayload) -> Episode:
        if not isinstance(payload, SerializedPayload):
            raise TypeError(f"Expected SerializedPayload, got {type(payload).__name__}")
        
        data = payload.data
        events: List[EmotionEvent] = []
        
        for event_dict in data['events']:
            events.append(EmotionEvent(
                timestamp=datetime.fromisoformat(event_dict['timestamp']),
                emotion_label=event_dict['emotion_label'],
                score=event_dict.get('score'),
                source_id=event_dict.get('source_id'),
                metadata=event_dict.get('metadata'),
            ))
        
        return Episode(
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            events=tuple(events),
            source_subject_id=data.get('source_subject_id'),
        )


class TemporalNarrativeGraphSerializer:
    
    @staticmethod
    def serialize(graph: TemporalNarrativeGraph) -> SerializedPayload:
        if not isinstance(graph, TemporalNarrativeGraph):
            raise TypeError(f"Expected TemporalNarrativeGraph, got {type(graph).__name__}")
        
        nodes_data: List[Dict[str, Any]] = []
        for node in graph.nodes:
            node_payload = EpisodeSerializer.serialize(node)
            nodes_data.append(node_payload.data)
        
        edges_data: List[Dict[str, Any]] = []
        for edge in graph.edges:
            edges_data.append({
                'source_index': edge.source_index,
                'target_index': edge.target_index,
                'relation': edge.relation.value,
            })
        
        data: Dict[str, Any] = {
            'nodes': nodes_data,
            'edges': edges_data,
        }
        if graph.adjacency_threshold is not None:
            data['adjacency_threshold_seconds'] = graph.adjacency_threshold.total_seconds()
        
        return SerializedPayload(
            data=data,
            schema_version=SCHEMA_VERSION,
            fingerprint=graph.graph_id,
        )
    
    @staticmethod
    def deserialize(payload: SerializedPayload) -> TemporalNarrativeGraph:
        if not isinstance(payload, SerializedPayload):
            raise TypeError(f"Expected SerializedPayload, got {type(payload).__name__}")
        
        data = payload.data
        
        nodes: List[Episode] = []
        for node_data in data['nodes']:
            node_payload = SerializedPayload(
                data=node_data,
                schema_version=payload.schema_version,
                fingerprint='',
            )
            nodes.append(EpisodeSerializer.deserialize(node_payload))
        
        edges: List[NarrativeEdge] = []
        for edge_data in data['edges']:
            edges.append(NarrativeEdge(
                source_index=edge_data['source_index'],
                target_index=edge_data['target_index'],
                relation=ProximityRelation(edge_data['relation']),
            ))
        
        adjacency_threshold: Optional[timedelta] = None
        if 'adjacency_threshold_seconds' in data:
            adjacency_threshold = timedelta(seconds=data['adjacency_threshold_seconds'])
        
        return TemporalNarrativeGraph(
            nodes=tuple(nodes),
            edges=tuple(edges),
            adjacency_threshold=adjacency_threshold,
        )
