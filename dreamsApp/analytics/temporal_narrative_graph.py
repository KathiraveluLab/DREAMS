# dreamsApp/analytics/temporal_narrative_graph.py

from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, List, Dict, Any, Optional

from .emotion_episode import Episode
from .episode_proximity import ProximityRelation, classify_episode_proximity


__all__ = [
    'NarrativeEdge',
    'TemporalNarrativeGraph',
    'build_narrative_graph',
]


@dataclass(frozen=True)
class NarrativeEdge:
    source_index: int
    target_index: int
    relation: ProximityRelation
    
    def __post_init__(self) -> None:
        if self.source_index < 0 or self.target_index < 0:
            raise ValueError(
                f"Indices must be non-negative: "
                f"source_index={self.source_index}, target_index={self.target_index}"
            )
        if self.source_index >= self.target_index:
            raise ValueError(
                f"source_index must be less than target_index for canonical ordering: "
                f"{self.source_index} >= {self.target_index}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_index': self.source_index,
            'target_index': self.target_index,
            'relation': self.relation.value,
        }


@dataclass(frozen=True)
class TemporalNarrativeGraph:
    nodes: Tuple[Episode, ...]
    edges: Tuple[NarrativeEdge, ...]
    adjacency_threshold: Optional[timedelta] = None
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def node_count(self) -> int:
        return len(self.nodes)
    
    def edge_count(self) -> int:
        return len(self.edges)
    
    def is_empty(self) -> bool:
        return len(self.nodes) == 0
    
    def edges_for_node(self, node_index: int) -> Tuple[NarrativeEdge, ...]:
        if node_index < 0 or node_index >= len(self.nodes):
            raise IndexError(f"node_index {node_index} out of bounds for graph with {len(self.nodes)} nodes")
        
        return tuple(
            edge for edge in self.edges
            if edge.source_index == node_index or edge.target_index == node_index
        )
    
    def edges_by_relation(self, relation: ProximityRelation) -> Tuple[NarrativeEdge, ...]:
        return tuple(edge for edge in self.edges if edge.relation == relation)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'adjacency_threshold_seconds': (
                self.adjacency_threshold.total_seconds()
                if self.adjacency_threshold is not None
                else None
            ),
            'node_count': self.node_count(),
            'edge_count': self.edge_count(),
        }


def build_narrative_graph(
    episodes: List[Episode],
    adjacency_threshold: timedelta = timedelta(0),
    include_disjoint_edges: bool = False
) -> TemporalNarrativeGraph:
    if not isinstance(episodes, list):
        raise TypeError(f"episodes must be a list, got {type(episodes).__name__}")
    if not isinstance(adjacency_threshold, timedelta):
        raise TypeError(f"adjacency_threshold must be a timedelta, got {type(adjacency_threshold).__name__}")
    if adjacency_threshold < timedelta(0):
        raise ValueError("adjacency_threshold must be non-negative")
    
    for i, episode in enumerate(episodes):
        if not isinstance(episode, Episode):
            raise TypeError(f"episodes[{i}] must be an Episode, got {type(episode).__name__}")
    
    if not episodes:
        return TemporalNarrativeGraph(
            nodes=(),
            edges=(),
            adjacency_threshold=adjacency_threshold
        )
    
    nodes = tuple(episodes)
    
    edges: List[NarrativeEdge] = []
    n = len(nodes)
    
    for i in range(n):
        for j in range(i + 1, n):
            relation = classify_episode_proximity(
                nodes[i],
                nodes[j],
                adjacency_threshold
            )
            
            if relation != ProximityRelation.DISJOINT or include_disjoint_edges:
                edge = NarrativeEdge(
                    source_index=i,
                    target_index=j,
                    relation=relation
                )
                edges.append(edge)
    
    return TemporalNarrativeGraph(
        nodes=nodes,
        edges=tuple(edges),
        adjacency_threshold=adjacency_threshold
    )
