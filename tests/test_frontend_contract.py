# tests/test_frontend_contract.py

import pytest
from datetime import datetime, timedelta

from dreamsApp.analytics.emotion_episode import Episode
from dreamsApp.analytics.temporal_narrative_graph import build_narrative_graph
from dreamsApp.analytics.frontend_contract import (
    FrontendNode,
    FrontendEdge,
    FrontendGraphPayload,
    build_frontend_payload,
)


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0)


class TestFrontendNodeStability:
    
    def test_node_id_matches_episode_id(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        graph = build_narrative_graph([episode])
        payload = build_frontend_payload(graph)
        
        assert payload.nodes[0].id == episode.episode_id
    
    def test_nodes_deterministic(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
            Episode(start_time=base_time + timedelta(hours=1), end_time=base_time + timedelta(hours=2), events=()),
        ]
        graph = build_narrative_graph(episodes)
        
        payload1 = build_frontend_payload(graph)
        payload2 = build_frontend_payload(graph)
        
        assert len(payload1.nodes) == len(payload2.nodes)
        for n1, n2 in zip(payload1.nodes, payload2.nodes):
            assert n1.id == n2.id
            assert n1.index == n2.index
            assert n1.start_time_iso == n2.start_time_iso


class TestFrontendEdgeStability:
    
    def test_edge_id_deterministic(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
            Episode(start_time=base_time + timedelta(hours=1), end_time=base_time + timedelta(hours=2), events=()),
        ]
        graph = build_narrative_graph(episodes, timedelta(0))
        
        payload1 = build_frontend_payload(graph)
        payload2 = build_frontend_payload(graph)
        
        assert len(payload1.edges) == len(payload2.edges)
        for e1, e2 in zip(payload1.edges, payload2.edges):
            assert e1.id == e2.id
            assert e1.source_id == e2.source_id
            assert e1.target_id == e2.target_id
    
    def test_edge_contains_node_ids(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
            Episode(start_time=base_time + timedelta(hours=1), end_time=base_time + timedelta(hours=2), events=()),
        ]
        graph = build_narrative_graph(episodes, timedelta(0))
        payload = build_frontend_payload(graph)
        
        assert payload.edge_count >= 1
        edge = payload.edges[0]
        
        node_ids = {n.id for n in payload.nodes}
        assert edge.source_id in node_ids
        assert edge.target_id in node_ids


class TestDeterministicOrdering:
    
    def test_nodes_sorted_by_start_time(self, base_time: datetime) -> None:
        ep3 = Episode(start_time=base_time + timedelta(hours=2), end_time=base_time + timedelta(hours=3), events=())
        ep1 = Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=())
        ep2 = Episode(start_time=base_time + timedelta(hours=1), end_time=base_time + timedelta(hours=2), events=())
        
        graph = build_narrative_graph([ep3, ep1, ep2])
        payload = build_frontend_payload(graph)
        
        start_times = [n.start_time_iso for n in payload.nodes]
        assert start_times == sorted(start_times)
    
    def test_node_indices_contiguous(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time + timedelta(hours=i), end_time=base_time + timedelta(hours=i + 1), events=())
            for i in range(5)
        ]
        graph = build_narrative_graph(episodes)
        payload = build_frontend_payload(graph)
        
        indices = [n.index for n in payload.nodes]
        assert indices == list(range(len(episodes)))
    
    def test_edges_sorted_by_source_target(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time + timedelta(hours=i), end_time=base_time + timedelta(hours=i + 1), events=())
            for i in range(3)
        ]
        graph = build_narrative_graph(episodes, timedelta(0))
        payload = build_frontend_payload(graph)
        
        edge_keys = [(e.source_index, e.target_index) for e in payload.edges]
        assert edge_keys == sorted(edge_keys)


class TestFrontendPayloadStructure:
    
    def test_payload_contains_required_fields(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
        ]
        graph = build_narrative_graph(episodes)
        payload = build_frontend_payload(graph)
        
        assert hasattr(payload, 'graph_id')
        assert hasattr(payload, 'nodes')
        assert hasattr(payload, 'edges')
        assert hasattr(payload, 'schema_version')
        assert hasattr(payload, 'node_count')
        assert hasattr(payload, 'edge_count')
    
    def test_to_dict(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
        ]
        graph = build_narrative_graph(episodes)
        payload = build_frontend_payload(graph)
        
        d = payload.to_dict()
        
        assert 'graph_id' in d
        assert 'nodes' in d
        assert 'edges' in d
        assert 'schema_version' in d
        assert d['node_count'] == 1
        assert d['edge_count'] == 0
    
    def test_node_to_dict(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        graph = build_narrative_graph([episode])
        payload = build_frontend_payload(graph)
        
        node_dict = payload.nodes[0].to_dict()
        
        assert 'id' in node_dict
        assert 'index' in node_dict
        assert 'start_time_iso' in node_dict
        assert 'end_time_iso' in node_dict
        assert 'event_count' in node_dict
        assert 'duration_seconds' in node_dict
    
    def test_empty_graph(self) -> None:
        graph = build_narrative_graph([])
        payload = build_frontend_payload(graph)
        
        assert payload.node_count == 0
        assert payload.edge_count == 0
        assert len(payload.nodes) == 0
        assert len(payload.edges) == 0


class TestPayloadImmutability:
    
    def test_payload_frozen(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        graph = build_narrative_graph([episode])
        payload = build_frontend_payload(graph)
        
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            payload.graph_id = "different"
    
    def test_node_frozen(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        graph = build_narrative_graph([episode])
        payload = build_frontend_payload(graph)
        
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            payload.nodes[0].id = "different"
