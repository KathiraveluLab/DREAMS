# tests/test_temporal_narrative_graph.py

import pytest
from datetime import datetime, timedelta
from dataclasses import FrozenInstanceError

from dreamsApp.analytics.emotion_timeline import EmotionEvent, EmotionTimeline
from dreamsApp.analytics.emotion_episode import Episode
from dreamsApp.analytics.episode_segmentation import segment_timeline_to_episodes
from dreamsApp.analytics.episode_proximity import (
    ProximityRelation,
    compute_temporal_overlap,
    compute_temporal_gap,
    are_episodes_adjacent,
    classify_episode_proximity,
)
from dreamsApp.analytics.temporal_narrative_graph import (
    NarrativeEdge,
    TemporalNarrativeGraph,
    build_narrative_graph,
)


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_events(base_time: datetime) -> tuple:
    return (
        EmotionEvent(timestamp=base_time, emotion_label="neutral"),
        EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="positive"),
        EmotionEvent(timestamp=base_time + timedelta(minutes=20), emotion_label="neutral"),
    )


@pytest.fixture
def sample_episode(base_time: datetime, sample_events: tuple) -> Episode:
    return Episode(
        start_time=base_time,
        end_time=base_time + timedelta(minutes=30),
        events=sample_events,
        source_subject_id="test_subject"
    )


class TestEpisodeImmutability:
    
    def test_episode_frozen(self, sample_episode: Episode):
        with pytest.raises(FrozenInstanceError):
            sample_episode.source_subject_id = "other"
    
    def test_episode_events_tuple(self, sample_episode: Episode):
        assert isinstance(sample_episode.events, tuple)


class TestEpisodeTemporalInvariants:
    
    def test_valid_episode_creation(self, base_time: datetime, sample_events: tuple):
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(minutes=30),
            events=sample_events
        )
        assert len(episode) == 3
    
    def test_start_after_end_raises(self, base_time: datetime):
        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            Episode(
                start_time=base_time + timedelta(hours=1),
                end_time=base_time,
                events=()
            )
    
    def test_event_before_start_raises(self, base_time: datetime):
        event = EmotionEvent(
            timestamp=base_time - timedelta(minutes=1),
            emotion_label="neutral"
        )
        with pytest.raises(ValueError, match="before episode start_time"):
            Episode(
                start_time=base_time,
                end_time=base_time + timedelta(hours=1),
                events=(event,)
            )
    
    def test_event_at_end_raises(self, base_time: datetime):
        event = EmotionEvent(
            timestamp=base_time + timedelta(hours=1),
            emotion_label="neutral"
        )
        with pytest.raises(ValueError, match="at or after episode end_time"):
            Episode(
                start_time=base_time,
                end_time=base_time + timedelta(hours=1),
                events=(event,)
            )
    
    def test_empty_episode_allowed(self, base_time: datetime):
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        assert episode.is_empty()
        assert len(episode) == 0


class TestEpisodeSegmentation:
    
    def test_empty_timeline_returns_empty(self):
        timeline = EmotionTimeline(subject_id="empty", events=())
        episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=5))
        assert episodes == []
    
    def test_single_segment_no_gaps(self, base_time: datetime):
        events = tuple(
            EmotionEvent(
                timestamp=base_time + timedelta(minutes=i),
                emotion_label="neutral"
            )
            for i in range(5)
        )
        timeline = EmotionTimeline(subject_id="test", events=events)
        episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=10))
        
        assert len(episodes) == 1
        assert len(episodes[0]) == 5
    
    def test_gap_creates_multiple_episodes(self, base_time: datetime):
        events = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=1), emotion_label="positive"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=31), emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=32), emotion_label="negative"),
        )
        timeline = EmotionTimeline(subject_id="test", events=events)
        episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=10))
        
        assert len(episodes) == 2
        assert len(episodes[0]) == 2
        assert len(episodes[1]) == 2
    
    def test_deterministic_segmentation(self, base_time: datetime):
        events = tuple(
            EmotionEvent(
                timestamp=base_time + timedelta(minutes=i * 5),
                emotion_label="neutral"
            )
            for i in range(10)
        )
        timeline = EmotionTimeline(subject_id="test", events=events)
        
        episodes1 = segment_timeline_to_episodes(timeline, timedelta(minutes=10))
        episodes2 = segment_timeline_to_episodes(timeline, timedelta(minutes=10))
        
        assert len(episodes1) == len(episodes2)
        for e1, e2 in zip(episodes1, episodes2):
            assert e1.start_time == e2.start_time
            assert e1.end_time == e2.end_time
            assert len(e1) == len(e2)


class TestEpisodeProximity:
    
    def test_overlap_identical_episodes(self, base_time: datetime):
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        overlap = compute_temporal_overlap(episode, episode)
        assert overlap == 1.0
    
    def test_overlap_no_overlap(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=2),
            end_time=base_time + timedelta(hours=3),
            events=()
        )
        overlap = compute_temporal_overlap(ep1, ep2)
        assert overlap == 0.0
    
    def test_overlap_partial(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=2),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=3),
            events=()
        )
        overlap = compute_temporal_overlap(ep1, ep2)
        assert 0.0 < overlap < 1.0
    
    def test_overlap_symmetric(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=2),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=3),
            events=()
        )
        assert compute_temporal_overlap(ep1, ep2) == compute_temporal_overlap(ep2, ep1)
    
    def test_gap_between_disjoint_episodes(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=2),
            end_time=base_time + timedelta(hours=3),
            events=()
        )
        gap = compute_temporal_gap(ep1, ep2)
        assert gap == 3600.0
    
    def test_adjacency_touching_episodes(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=2),
            events=()
        )
        assert are_episodes_adjacent(ep1, ep2, timedelta(0))
    
    def test_classification_overlapping(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=2),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=3),
            events=()
        )
        relation = classify_episode_proximity(ep1, ep2)
        assert relation == ProximityRelation.OVERLAPPING
    
    def test_classification_adjacent(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=2),
            events=()
        )
        relation = classify_episode_proximity(ep1, ep2, timedelta(0))
        assert relation == ProximityRelation.ADJACENT
    
    def test_classification_disjoint(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=5),
            end_time=base_time + timedelta(hours=6),
            events=()
        )
        relation = classify_episode_proximity(ep1, ep2, timedelta(hours=1))
        assert relation == ProximityRelation.DISJOINT
    
    def test_classification_symmetric(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=2),
            end_time=base_time + timedelta(hours=3),
            events=()
        )
        assert classify_episode_proximity(ep1, ep2) == classify_episode_proximity(ep2, ep1)


class TestNarrativeEdge:
    
    def test_valid_edge(self):
        edge = NarrativeEdge(
            source_index=0,
            target_index=1,
            relation=ProximityRelation.ADJACENT
        )
        assert edge.source_index == 0
        assert edge.target_index == 1
    
    def test_invalid_ordering_raises(self):
        with pytest.raises(ValueError, match="source_index must be less than target_index"):
            NarrativeEdge(
                source_index=1,
                target_index=0,
                relation=ProximityRelation.ADJACENT
            )
    
    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            NarrativeEdge(
                source_index=-1,
                target_index=1,
                relation=ProximityRelation.ADJACENT
            )


class TestTemporalNarrativeGraph:
    
    def test_empty_graph(self):
        graph = build_narrative_graph([])
        assert graph.is_empty()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
    
    def test_single_episode_graph(self, base_time: datetime):
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        graph = build_narrative_graph([episode])
        
        assert graph.node_count() == 1
        assert graph.edge_count() == 0
    
    def test_adjacent_episodes_create_edge(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=2),
            events=()
        )
        graph = build_narrative_graph([ep1, ep2], timedelta(0))
        
        assert graph.node_count() == 2
        assert graph.edge_count() == 1
        assert graph.edges[0].relation == ProximityRelation.ADJACENT
    
    def test_disjoint_edges_excluded_by_default(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=10),
            end_time=base_time + timedelta(hours=11),
            events=()
        )
        graph = build_narrative_graph([ep1, ep2])
        
        assert graph.node_count() == 2
        assert graph.edge_count() == 0
    
    def test_include_disjoint_edges(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=10),
            end_time=base_time + timedelta(hours=11),
            events=()
        )
        graph = build_narrative_graph([ep1, ep2], include_disjoint_edges=True)
        
        assert graph.edge_count() == 1
        assert graph.edges[0].relation == ProximityRelation.DISJOINT
    
    def test_graph_immutability(self, base_time: datetime):
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=()
        )
        graph = build_narrative_graph([episode])
        
        with pytest.raises(FrozenInstanceError):
            graph.adjacency_threshold = timedelta(hours=5)
    
    def test_deterministic_construction(self, base_time: datetime):
        episodes = [
            Episode(
                start_time=base_time + timedelta(hours=i),
                end_time=base_time + timedelta(hours=i + 1),
                events=()
            )
            for i in range(3)
        ]
        
        graph1 = build_narrative_graph(episodes, timedelta(0))
        graph2 = build_narrative_graph(episodes, timedelta(0))
        
        assert graph1.node_count() == graph2.node_count()
        assert graph1.edge_count() == graph2.edge_count()
        
        for e1, e2 in zip(graph1.edges, graph2.edges):
            assert e1.source_index == e2.source_index
            assert e1.target_index == e2.target_index
            assert e1.relation == e2.relation
    
    def test_edges_for_node(self, base_time: datetime):
        episodes = [
            Episode(
                start_time=base_time + timedelta(hours=i),
                end_time=base_time + timedelta(hours=i + 1),
                events=()
            )
            for i in range(3)
        ]
        graph = build_narrative_graph(episodes, timedelta(0))
        
        middle_edges = graph.edges_for_node(1)
        assert len(middle_edges) == 2
    
    def test_edges_by_relation(self, base_time: datetime):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=2),
            events=()
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=3),
            events=()
        )
        ep3 = Episode(
            start_time=base_time + timedelta(hours=3),
            end_time=base_time + timedelta(hours=4),
            events=()
        )
        graph = build_narrative_graph([ep1, ep2, ep3], timedelta(0))
        
        overlapping = graph.edges_by_relation(ProximityRelation.OVERLAPPING)
        adjacent = graph.edges_by_relation(ProximityRelation.ADJACENT)
        
        assert len(overlapping) >= 1
        assert all(e.relation == ProximityRelation.OVERLAPPING for e in overlapping)
