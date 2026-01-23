# tests/test_serialization.py

import pytest
from datetime import datetime, timedelta

from dreamsApp.analytics.emotion_timeline import EmotionEvent, EmotionTimeline
from dreamsApp.analytics.emotion_episode import Episode
from dreamsApp.analytics.temporal_narrative_graph import build_narrative_graph
from dreamsApp.analytics.episode_proximity import ProximityRelation
from dreamsApp.analytics.serialization import (
    SCHEMA_VERSION,
    SerializedPayload,
    EmotionTimelineSerializer,
    EpisodeSerializer,
    TemporalNarrativeGraphSerializer,
)


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0)


class TestSerializedPayload:
    
    def test_to_json_deterministic(self) -> None:
        payload = SerializedPayload(
            data={'b': 2, 'a': 1},
            schema_version='1.0',
            fingerprint='abc123'
        )
        json1 = payload.to_json()
        json2 = payload.to_json()
        assert json1 == json2
    
    def test_to_json_sorted_keys(self) -> None:
        payload = SerializedPayload(
            data={'z': 1, 'a': 2},
            schema_version='1.0',
            fingerprint='abc'
        )
        json_str = payload.to_json()
        assert '"a":2' in json_str
        assert json_str.index('"a"') < json_str.index('"z"')
    
    def test_round_trip(self) -> None:
        payload = SerializedPayload(
            data={'key': 'value', 'number': 42},
            schema_version='1.0',
            fingerprint='test123'
        )
        json_str = payload.to_json()
        restored = SerializedPayload.from_json(json_str)
        
        assert restored.data == payload.data
        assert restored.schema_version == payload.schema_version
        assert restored.fingerprint == payload.fingerprint


class TestEmotionTimelineSerializer:
    
    def test_serialize_basic(self, base_time: datetime) -> None:
        events = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
        )
        timeline = EmotionTimeline(subject_id="test", events=events)
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        assert payload.schema_version == SCHEMA_VERSION
        assert payload.fingerprint == timeline.fingerprint()
        assert payload.data['subject_id'] == "test"
        assert len(payload.data['events']) == 1
    
    def test_serialize_with_optional_fields(self, base_time: datetime) -> None:
        events = (
            EmotionEvent(
                timestamp=base_time,
                emotion_label="happy",
                score=0.95,
                source_id="src1",
                metadata={"key": "value"}
            ),
        )
        timeline = EmotionTimeline(subject_id="test", events=events)
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        event_data = payload.data['events'][0]
        assert event_data['score'] == 0.95
        assert event_data['source_id'] == "src1"
        assert event_data['metadata'] == {"key": "value"}
    
    def test_round_trip(self, base_time: datetime) -> None:
        events = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral", score=0.5),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="happy"),
        )
        timeline = EmotionTimeline(subject_id="test", events=events)
        
        payload = EmotionTimelineSerializer.serialize(timeline)
        restored = EmotionTimelineSerializer.deserialize(payload)
        
        assert restored.subject_id == timeline.subject_id
        assert len(restored.events) == len(timeline.events)
        assert restored.events[0].timestamp == timeline.events[0].timestamp
        assert restored.events[0].emotion_label == timeline.events[0].emotion_label
        assert restored.events[0].score == timeline.events[0].score
        assert restored.fingerprint() == timeline.fingerprint()
    
    def test_round_trip_empty(self) -> None:
        timeline = EmotionTimeline(subject_id="empty", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        restored = EmotionTimelineSerializer.deserialize(payload)
        
        assert restored.subject_id == timeline.subject_id
        assert len(restored.events) == 0


class TestEpisodeSerializer:
    
    def test_serialize_basic(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        payload = EpisodeSerializer.serialize(episode)
        
        assert payload.schema_version == SCHEMA_VERSION
        assert payload.fingerprint == episode.episode_id
        assert 'start_time' in payload.data
        assert 'end_time' in payload.data
    
    def test_round_trip(self, base_time: datetime) -> None:
        events = (
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="neutral"),
        )
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=events,
            source_subject_id="subj1"
        )
        
        payload = EpisodeSerializer.serialize(episode)
        restored = EpisodeSerializer.deserialize(payload)
        
        assert restored.start_time == episode.start_time
        assert restored.end_time == episode.end_time
        assert restored.source_subject_id == episode.source_subject_id
        assert len(restored.events) == len(episode.events)
        assert restored.episode_id == episode.episode_id
    
    def test_round_trip_no_subject_id(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        payload = EpisodeSerializer.serialize(episode)
        restored = EpisodeSerializer.deserialize(payload)
        
        assert restored.source_subject_id is None


class TestTemporalNarrativeGraphSerializer:
    
    def test_serialize_empty_graph(self) -> None:
        graph = build_narrative_graph([])
        payload = TemporalNarrativeGraphSerializer.serialize(graph)
        
        assert payload.schema_version == SCHEMA_VERSION
        assert payload.data['nodes'] == []
        assert payload.data['edges'] == []
    
    def test_round_trip(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
            Episode(start_time=base_time + timedelta(hours=1), end_time=base_time + timedelta(hours=2), events=()),
        ]
        graph = build_narrative_graph(episodes, timedelta(0))
        
        payload = TemporalNarrativeGraphSerializer.serialize(graph)
        restored = TemporalNarrativeGraphSerializer.deserialize(payload)
        
        assert restored.node_count() == graph.node_count()
        assert restored.edge_count() == graph.edge_count()
        assert restored.graph_id == graph.graph_id
        
        for orig_node, rest_node in zip(graph.nodes, restored.nodes):
            assert orig_node.episode_id == rest_node.episode_id
        
        for orig_edge, rest_edge in zip(graph.edges, restored.edges):
            assert orig_edge.source_index == rest_edge.source_index
            assert orig_edge.target_index == rest_edge.target_index
            assert orig_edge.relation == rest_edge.relation
    
    def test_adjacency_threshold_preserved(self, base_time: datetime) -> None:
        threshold = timedelta(minutes=30)
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
        ]
        graph = build_narrative_graph(episodes, threshold)
        
        payload = TemporalNarrativeGraphSerializer.serialize(graph)
        restored = TemporalNarrativeGraphSerializer.deserialize(payload)
        
        assert restored.adjacency_threshold == threshold
