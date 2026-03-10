# dreamsApp/analytics/temporal_narrative_graph.py

from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, List, Dict, Any, Optional

from .emotion_episode import Episode
from .episode_proximity import (
    ProximityRelation,
    classify_episode_proximity,
    compute_temporal_gap,
    compute_temporal_overlap,
)


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
    weight: float = 1.0
    
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
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(
                f"weight must be between 0.0 and 1.0, got {self.weight}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_index': self.source_index,
            'target_index': self.target_index,
            'relation': self.relation.value,
            'weight': self.weight,
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
    
    def to_networkx(self) -> Any:
        """
        Convert to a networkx directed graph for graph-theoretic analysis.

        Each node is identified by its integer index (0..N-1).
        Node attributes:
            - start_time: datetime (from the Episode)
            - end_time: datetime (from the Episode)
            - emotion_label: str — the dominant (most frequent) emotion_label
              among the Episode's events; ``None`` if episode has no events.
            - event_count: int — number of EmotionEvents in the episode.

        Edge attributes:
            - relation: str — the ProximityRelation.value
              ('overlapping', 'adjacent', 'disjoint').
            - weight: float — connection strength (0.0–1.0).

        Returns:
            networkx.DiGraph
        """
        import networkx as nx  # lazy import to avoid hard module-level dep
        from collections import Counter
        
        G = nx.DiGraph()
        
        for i, episode in enumerate(self.nodes):
            if episode.events:
                label_counts = Counter(
                    e.emotion_label for e in episode.events
                )
                dominant_label = label_counts.most_common(1)[0][0]
            else:
                dominant_label = None
            
            G.add_node(
                i,
                start_time=episode.start_time,
                end_time=episode.end_time,
                emotion_label=dominant_label,
                event_count=len(episode.events),
            )
        
        for edge in self.edges:
            G.add_edge(
                edge.source_index,
                edge.target_index,
                relation=edge.relation.value,
                weight=edge.weight,
            )
        
        return G
    
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
                weight = _compute_edge_weight(
                    nodes[i], nodes[j], relation, adjacency_threshold
                )
                edge = NarrativeEdge(
                    source_index=i,
                    target_index=j,
                    relation=relation,
                    weight=weight,
                )
                edges.append(edge)
    
    return TemporalNarrativeGraph(
        nodes=nodes,
        edges=tuple(edges),
        adjacency_threshold=adjacency_threshold
    )


def _compute_edge_weight(
    ep_a: Episode,
    ep_b: Episode,
    relation: ProximityRelation,
    adjacency_threshold: timedelta,
) -> float:
    """Compute a 0.0–1.0 connection-strength weight for an edge.

    - **Overlapping** episodes get weight 1.0 (strongest possible link).
    - **Adjacent** episodes are scored as ``1 - gap / threshold``, giving
      1.0 for touching episodes and approaching 0.0 as the gap nears the
      threshold.
    - **Disjoint** edges (only present when *include_disjoint_edges* is
      True) always receive 0.0.
    """
    if relation == ProximityRelation.OVERLAPPING:
        return 1.0

    if relation == ProximityRelation.DISJOINT:
        return 0.0

    # Adjacent — weight decreases linearly with the gap.
    threshold_secs = adjacency_threshold.total_seconds()
    if threshold_secs == 0:
        # Zero threshold means only touching episodes qualify; gap is 0.
        return 1.0

    gap_secs = compute_temporal_gap(ep_a, ep_b)
    return max(0.0, 1.0 - gap_secs / threshold_secs)
