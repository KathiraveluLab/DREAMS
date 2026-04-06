# dreamsApp/analytics/graph_analysis.py

"""
Structural Graph Metrics for Temporal Narrative Graphs

Computes quantitative metrics from a TemporalNarrativeGraph using graph theory.
This module is purely analytical — no Flask, no database, no side effects.

Computed metrics:
    - Graph-level statistics (density, components, DAG check)
    - Node-level centrality measures (degree, betweenness)
    - Pattern analysis (common transitions, emotional cycles, label distribution)

Known limitations:
    - The current sentiment model produces only 'positive', 'negative', 'neutral'.
      Transition and cycle analysis is limited to these 3 labels. When finer-grained
      models (e.g., CHIME dimensions) are integrated, the analysis will automatically
      benefit without code changes.
    - NarrativeEdge enforces source_index < target_index, so the DiGraph is always
      a DAG. Structural cycle detection on the DiGraph returns empty. Cycles are
      therefore computed on a label-level graph instead.
"""

from collections import Counter
from typing import Dict, Any, List
import copy
import networkx as nx

from .temporal_narrative_graph import TemporalNarrativeGraph


__all__ = ['analyze_narrative_graph']


_EMPTY_RESPONSE: Dict[str, Any] = {
    "graph_summary": {
        "node_count": 0,
        "edge_count": 0,
        "density": 0.0,
        "connected_components": 0,
        "is_dag": True,
    },
    "node_metrics": [],
    "edges": [],
    "pattern_analysis": {
        "common_transitions": [],
        "emotional_cycles": [],
        "label_distribution": {},
    },
}

_MAX_TRANSITIONS = 10
_MAX_CYCLES = 20


def analyze_narrative_graph(graph: TemporalNarrativeGraph) -> Dict[str, Any]:
    """
    Compute structural metrics from a TemporalNarrativeGraph.

    Args:
        graph: A TemporalNarrativeGraph instance.

    Returns:
        A JSON-serializable dict with keys:
            - graph_summary: graph-level statistics
            - node_metrics: per-node centrality and metadata
            - pattern_analysis: transitions, cycles, label counts

    Raises:
        TypeError: if *graph* is not a TemporalNarrativeGraph.
    """
    if not isinstance(graph, TemporalNarrativeGraph):
        raise TypeError(
            f"graph must be a TemporalNarrativeGraph, got {type(graph).__name__}"
        )

    if graph.is_empty():
        # Return a deep copy so callers cannot mutate the sentinel.
        return copy.deepcopy(_EMPTY_RESPONSE)

    G = graph.to_networkx()

    graph_summary = _compute_graph_summary(G)
    node_metrics = _compute_node_metrics(G)
    pattern_analysis = _compute_pattern_analysis(G)
    edges = _compute_edges(G)

    return {
        "graph_summary": graph_summary,
        "node_metrics": node_metrics,
        "pattern_analysis": pattern_analysis,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Internal helpers — not exported
# ---------------------------------------------------------------------------


def _compute_edges(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Return the raw directed edge list with source/target node indices."""
    edges: List[Dict[str, Any]] = []
    for u, v, attrs in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "relation": attrs.get("relation", ""),
            "weight": round(attrs.get("weight", 1.0), 4),
        })
    return edges


def _compute_graph_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """Graph-level statistics."""
    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "density": nx.density(G),
        "connected_components": nx.number_weakly_connected_components(G),
        "is_dag": nx.is_directed_acyclic_graph(G),
    }


def _compute_node_metrics(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Per-node centrality and metadata, ordered by node index."""
    # Pre-compute centrality dicts once (O(V+E) or O(V*E) depending on metric).
    degree_cent = nx.degree_centrality(G)
    in_degree_cent = nx.in_degree_centrality(G)
    out_degree_cent = nx.out_degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, weight='weight')

    metrics: List[Dict[str, Any]] = []
    for i in sorted(G.nodes()):
        attrs = G.nodes[i]
        metrics.append({
            "node_index": i,
            "emotion_label": attrs.get("emotion_label"),
            "start_time": attrs["start_time"].isoformat(),
            "end_time": attrs["end_time"].isoformat(),
            "event_count": attrs.get("event_count", 0),
            "degree_centrality": degree_cent.get(i, 0.0),
            "in_degree_centrality": in_degree_cent.get(i, 0.0),
            "out_degree_centrality": out_degree_cent.get(i, 0.0),
            "betweenness_centrality": betweenness_cent.get(i, 0.0),
        })
    return metrics


def _compute_pattern_analysis(G: nx.DiGraph) -> Dict[str, Any]:
    """Transitions, emotional cycles, and label distribution."""
    return {
        "common_transitions": _compute_common_transitions(G),
        "emotional_cycles": _compute_emotional_cycles(G),
        "label_distribution": _compute_label_distribution(G),
    }


def _compute_common_transitions(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Most frequent emotion-to-emotion transitions, sorted by count desc."""
    transition_counter: Counter = Counter()
    for u, v in G.edges():
        src_label = G.nodes[u].get("emotion_label")
        tgt_label = G.nodes[v].get("emotion_label")
        if src_label is not None and tgt_label is not None:
            transition_counter[(src_label, tgt_label)] += 1

    top = transition_counter.most_common(_MAX_TRANSITIONS)
    return [
        {
            "source_emotion": pair[0],
            "target_emotion": pair[1],
            "count": count,
        }
        for pair, count in top
    ]


def _compute_emotional_cycles(G: nx.DiGraph) -> List[List[str]]:
    """
    Detect recurring emotional loops at the **label level**.

    The structural DiGraph is a DAG by construction (source_index < target_index),
    so it never contains structural cycles.  Instead, we build a label-level
    directed graph and find simple cycles there.  For example, if the original
    graph has edges negative→positive and positive→negative (across different
    episodes), the label-level graph will contain the cycle [negative, positive].
    """
    label_graph = nx.DiGraph()
    for u, v in G.edges():
        src = G.nodes[u].get("emotion_label")
        tgt = G.nodes[v].get("emotion_label")
        if src is None or tgt is None:
            continue
        if label_graph.has_edge(src, tgt):
            label_graph[src][tgt]["weight"] += 1
        else:
            label_graph.add_edge(src, tgt, weight=1)

    # Sort cycles by length to prioritize shorter, more fundamental loops.
    all_cycles = sorted(nx.simple_cycles(label_graph), key=len)
    return all_cycles[:_MAX_CYCLES]


def _compute_label_distribution(G: nx.DiGraph) -> Dict[str, int]:
    """Count of each emotion label across all nodes."""
    counter: Counter = Counter()
    for _, attrs in G.nodes(data=True):
        label = attrs.get("emotion_label")
        if label is not None:
            counter[label] += 1
    return dict(counter)
