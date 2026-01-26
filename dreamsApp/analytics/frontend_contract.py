# dreamsApp/analytics/frontend_contract.py

import hashlib
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from .emotion_episode import Episode
from .episode_proximity import ProximityRelation
from .temporal_narrative_graph import NarrativeEdge, TemporalNarrativeGraph
from .serialization import SCHEMA_VERSION


__all__ = [
    'FrontendNode',
    'FrontendEdge',
    'FrontendGraphPayload',
    'build_frontend_payload',
]


@dataclass(frozen=True)
class FrontendNode:
    id: str
    index: int
    start_time_iso: str
    end_time_iso: str
    event_count: int
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'index': self.index,
            'start_time_iso': self.start_time_iso,
            'end_time_iso': self.end_time_iso,
            'event_count': self.event_count,
            'duration_seconds': self.duration_seconds,
        }


@dataclass(frozen=True)
class FrontendEdge:
    id: str
    source_id: str
    target_id: str
    source_index: int
    target_index: int
    relation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'source_index': self.source_index,
            'target_index': self.target_index,
            'relation': self.relation,
        }


def _compute_edge_id(source_id: str, target_id: str) -> str:
    edge_str = f"{source_id}:{target_id}"
    return hashlib.sha256(edge_str.encode('utf-8')).hexdigest()[:32]


@dataclass(frozen=True)
class FrontendGraphPayload:
    graph_id: str
    nodes: Tuple[FrontendNode, ...]
    edges: Tuple[FrontendEdge, ...]
    schema_version: str
    node_count: int
    edge_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'graph_id': self.graph_id,
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'schema_version': self.schema_version,
            'node_count': self.node_count,
            'edge_count': self.edge_count,
        }


def build_frontend_payload(graph: TemporalNarrativeGraph) -> FrontendGraphPayload:
    if not isinstance(graph, TemporalNarrativeGraph):
        raise TypeError(f"Expected TemporalNarrativeGraph, got {type(graph).__name__}")
    
    sorted_nodes = sorted(
        enumerate(graph.nodes),
        key=lambda x: x[1].start_time
    )
    
    old_to_new_index = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_nodes)}
    
    frontend_nodes = []
    for new_idx, (old_idx, episode) in enumerate(sorted_nodes):
        frontend_nodes.append(FrontendNode(
            id=episode.episode_id,
            index=new_idx,
            start_time_iso=episode.start_time.isoformat(),
            end_time_iso=episode.end_time.isoformat(),
            event_count=len(episode),
            duration_seconds=episode.duration(),
        ))
    
    frontend_edges = []
    for edge in graph.edges:
        source_new_idx = old_to_new_index[edge.source_index]
        target_new_idx = old_to_new_index[edge.target_index]
        
        source_episode = graph.nodes[edge.source_index]
        target_episode = graph.nodes[edge.target_index]
        
        if source_new_idx > target_new_idx:
            source_new_idx, target_new_idx = target_new_idx, source_new_idx
            source_episode, target_episode = target_episode, source_episode
        
        edge_id = _compute_edge_id(source_episode.episode_id, target_episode.episode_id)
        
        frontend_edges.append(FrontendEdge(
            id=edge_id,
            source_id=source_episode.episode_id,
            target_id=target_episode.episode_id,
            source_index=source_new_idx,
            target_index=target_new_idx,
            relation=edge.relation.value,
        ))
    
    frontend_edges_sorted = sorted(
        frontend_edges,
        key=lambda e: (e.source_index, e.target_index)
    )
    
    return FrontendGraphPayload(
        graph_id=graph.graph_id,
        nodes=tuple(frontend_nodes),
        edges=tuple(frontend_edges_sorted),
        schema_version=SCHEMA_VERSION,
        node_count=len(frontend_nodes),
        edge_count=len(frontend_edges_sorted),
    )
