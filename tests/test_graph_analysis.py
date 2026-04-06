# tests/test_graph_analysis.py

"""
Unit tests for dreamsApp.core.graph.graph_analysis and
TemporalNarrativeGraph.to_networkx().

No external services (Flask, MongoDB) are needed — these tests exercise
the pure-analysis layer only.
"""

import pytest
from datetime import datetime, timedelta

from dreamsApp.core.graph.emotion_timeline import EmotionEvent
from dreamsApp.core.graph.emotion_episode import Episode
from dreamsApp.core.graph.temporal_narrative_graph import (
    TemporalNarrativeGraph,
    build_narrative_graph,
)
from dreamsApp.core.graph.graph_analysis import analyze_narrative_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_episode(base, hour_offset, duration_hours, labels):
    """Create an Episode with events carrying the given emotion labels.

    Events are spaced 5 minutes apart starting from *base + hour_offset*.
    """
    start = base + timedelta(hours=hour_offset)
    end = start + timedelta(hours=duration_hours)
    events = tuple(
        EmotionEvent(
            timestamp=start + timedelta(minutes=i * 5),
            emotion_label=label,
        )
        for i, label in enumerate(labels)
    )
    return Episode(start_time=start, end_time=end, events=events)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# 1.  Input Validation
# ---------------------------------------------------------------------------

class TestAnalyzeInputValidation:

    def test_non_graph_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a TemporalNarrativeGraph"):
            analyze_narrative_graph("not a graph")

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            analyze_narrative_graph(None)

    def test_empty_graph_returns_zero_metrics(self):
        graph = build_narrative_graph([])
        result = analyze_narrative_graph(graph)

        assert result["graph_summary"]["node_count"] == 0
        assert result["graph_summary"]["edge_count"] == 0
        assert result["graph_summary"]["density"] == 0.0
        assert result["graph_summary"]["connected_components"] == 0
        assert result["graph_summary"]["is_dag"] is True
        assert result["node_metrics"] == []
        assert result["pattern_analysis"]["common_transitions"] == []
        assert result["pattern_analysis"]["emotional_cycles"] == []
        assert result["pattern_analysis"]["label_distribution"] == {}

    def test_empty_graph_returns_fresh_copy(self):
        """Mutating one result must not affect subsequent calls."""
        graph = build_narrative_graph([])
        r1 = analyze_narrative_graph(graph)
        r1["node_metrics"].append("junk")
        r2 = analyze_narrative_graph(graph)
        assert r2["node_metrics"] == []


# ---------------------------------------------------------------------------
# 2.  Graph Summary
# ---------------------------------------------------------------------------

class TestGraphSummary:

    def test_single_node_graph(self, base_time):
        ep = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        result = analyze_narrative_graph(build_narrative_graph([ep]))
        summary = result["graph_summary"]

        assert summary["node_count"] == 1
        assert summary["edge_count"] == 0
        assert summary["density"] == 0.0
        assert summary["connected_components"] == 1
        assert summary["is_dag"] is True

    def test_two_adjacent_episodes(self, base_time):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=2),
            events=(),
        )
        result = analyze_narrative_graph(
            build_narrative_graph([ep1, ep2], adjacency_threshold=timedelta(0))
        )
        summary = result["graph_summary"]

        assert summary["node_count"] == 2
        assert summary["edge_count"] == 1
        assert summary["density"] > 0
        assert summary["connected_components"] == 1

    def test_disconnected_episodes(self, base_time):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=100),
            end_time=base_time + timedelta(hours=101),
            events=(),
        )
        result = analyze_narrative_graph(build_narrative_graph([ep1, ep2]))
        assert result["graph_summary"]["connected_components"] == 2

    def test_density_increases_with_edges(self, base_time):
        # Sparse: two disjoint episodes (0 edges)
        sparse = build_narrative_graph([
            Episode(start_time=base_time,
                    end_time=base_time + timedelta(hours=1), events=()),
            Episode(start_time=base_time + timedelta(hours=100),
                    end_time=base_time + timedelta(hours=101), events=()),
        ])
        # Dense: two overlapping episodes (1 edge)
        dense = build_narrative_graph([
            Episode(start_time=base_time,
                    end_time=base_time + timedelta(hours=2), events=()),
            Episode(start_time=base_time + timedelta(hours=1),
                    end_time=base_time + timedelta(hours=3), events=()),
        ])
        d_sparse = analyze_narrative_graph(sparse)["graph_summary"]["density"]
        d_dense = analyze_narrative_graph(dense)["graph_summary"]["density"]
        assert d_dense > d_sparse


# ---------------------------------------------------------------------------
# 3.  Node Metrics
# ---------------------------------------------------------------------------

class TestNodeMetrics:

    def test_node_metrics_count_matches_nodes(self, base_time):
        eps = [
            Episode(start_time=base_time + timedelta(hours=i),
                    end_time=base_time + timedelta(hours=i + 1), events=())
            for i in range(4)
        ]
        graph = build_narrative_graph(eps, adjacency_threshold=timedelta(0))
        result = analyze_narrative_graph(graph)
        assert len(result["node_metrics"]) == graph.node_count()

    def test_node_metrics_contain_required_keys(self, base_time):
        ep = make_episode(base_time, 0, 2, ["positive"])
        result = analyze_narrative_graph(build_narrative_graph([ep]))
        required = {
            "node_index", "emotion_label", "start_time", "end_time",
            "event_count", "degree_centrality", "in_degree_centrality",
            "out_degree_centrality", "betweenness_centrality",
        }
        for node in result["node_metrics"]:
            assert required.issubset(node.keys())

    def test_central_node_has_higher_betweenness(self, base_time):
        """Middle episode adjacent to both ends should have highest betweenness."""
        eps = [
            Episode(start_time=base_time + timedelta(hours=i),
                    end_time=base_time + timedelta(hours=i + 1), events=())
            for i in range(3)
        ]
        result = analyze_narrative_graph(
            build_narrative_graph(eps, adjacency_threshold=timedelta(0))
        )
        betweenness = [n["betweenness_centrality"] for n in result["node_metrics"]]
        # Node 1 sits between 0 and 2
        assert betweenness[1] >= betweenness[0]
        assert betweenness[1] >= betweenness[2]

    def test_emotion_label_derived_from_majority(self, base_time):
        ep = make_episode(base_time, 0, 2, ["negative", "negative", "positive"])
        result = analyze_narrative_graph(build_narrative_graph([ep]))
        assert result["node_metrics"][0]["emotion_label"] == "negative"

    def test_empty_episode_emotion_label_is_none(self, base_time):
        ep = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        result = analyze_narrative_graph(build_narrative_graph([ep]))
        assert result["node_metrics"][0]["emotion_label"] is None

    def test_timestamps_are_iso_strings(self, base_time):
        ep = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        result = analyze_narrative_graph(build_narrative_graph([ep]))
        node = result["node_metrics"][0]
        # Must be parseable ISO strings, not datetime objects
        assert isinstance(node["start_time"], str)
        assert isinstance(node["end_time"], str)
        datetime.fromisoformat(node["start_time"])
        datetime.fromisoformat(node["end_time"])


# ---------------------------------------------------------------------------
# 4.  Pattern Analysis
# ---------------------------------------------------------------------------

class TestPatternAnalysis:

    def test_common_transitions_format(self, base_time):
        ep1 = make_episode(base_time, 0, 2, ["negative"])
        ep2 = make_episode(base_time, 1, 2, ["positive"])
        graph = build_narrative_graph([ep1, ep2])
        result = analyze_narrative_graph(graph)

        for t in result["pattern_analysis"]["common_transitions"]:
            assert "source_emotion" in t
            assert "target_emotion" in t
            assert "count" in t
            assert isinstance(t["count"], int)

    def test_common_transitions_sorted_by_count(self, base_time):
        # 3 overlapping episodes: neg→pos appears 2x, neg→neu 1x, pos→neu 1x
        ep1 = make_episode(base_time, 0, 3, ["negative"])
        ep2 = make_episode(base_time, 1, 3, ["positive"])
        ep3 = make_episode(base_time, 2, 3, ["neutral"])
        graph = build_narrative_graph([ep1, ep2, ep3])
        transitions = analyze_narrative_graph(graph)["pattern_analysis"]["common_transitions"]
        counts = [t["count"] for t in transitions]
        assert counts == sorted(counts, reverse=True)

    def test_emotional_cycles_detected(self, base_time):
        """Episodes: neg→pos→neg should produce a label-level cycle."""
        ep1 = make_episode(base_time, 0, 2, ["negative"])
        ep2 = make_episode(base_time, 1, 2, ["positive"])
        ep3 = make_episode(base_time, 2, 2, ["negative"])
        graph = build_narrative_graph([ep1, ep2, ep3])
        cycles = analyze_narrative_graph(graph)["pattern_analysis"]["emotional_cycles"]

        # There should be a cycle containing both "negative" and "positive"
        cycle_sets = [set(c) for c in cycles]
        assert any({"negative", "positive"}.issubset(s) for s in cycle_sets)

    def test_no_cycles_for_single_label_no_self_loop(self, base_time):
        """Two disjoint episodes with the same label produce no edges,
        hence no label-level cycle."""
        ep1 = make_episode(base_time, 0, 1, ["negative"])
        ep2 = make_episode(base_time, 100, 1, ["negative"])
        graph = build_narrative_graph([ep1, ep2])
        cycles = analyze_narrative_graph(graph)["pattern_analysis"]["emotional_cycles"]
        assert cycles == []

    def test_self_loop_cycle_same_label(self, base_time):
        """Two adjacent episodes with the same label produce a self-loop
        in the label graph, which is a cycle of length 1."""
        ep1 = make_episode(base_time, 0, 1, ["negative"])
        ep2 = make_episode(base_time, 1, 1, ["negative"])
        graph = build_narrative_graph(
            [ep1, ep2], adjacency_threshold=timedelta(0)
        )
        cycles = analyze_narrative_graph(graph)["pattern_analysis"]["emotional_cycles"]
        assert any(len(c) == 1 and c[0] == "negative" for c in cycles)

    def test_label_distribution(self, base_time):
        ep1 = make_episode(base_time, 0, 2, ["negative", "negative"])
        ep2 = make_episode(base_time, 1, 2, ["positive"])
        ep3 = make_episode(base_time, 2, 2, ["neutral"])
        graph = build_narrative_graph([ep1, ep2, ep3])
        dist = analyze_narrative_graph(graph)["pattern_analysis"]["label_distribution"]

        assert dist["negative"] == 1  # ep1 dominant label
        assert dist["positive"] == 1
        assert dist["neutral"] == 1

    def test_label_distribution_skips_none(self, base_time):
        ep = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        dist = analyze_narrative_graph(
            build_narrative_graph([ep])
        )["pattern_analysis"]["label_distribution"]
        assert dist == {}


# ---------------------------------------------------------------------------
# 5.  to_networkx() conversion
# ---------------------------------------------------------------------------

class TestToNetworkx:

    def test_networkx_node_count(self, base_time):
        eps = [
            Episode(start_time=base_time + timedelta(hours=i),
                    end_time=base_time + timedelta(hours=i + 1), events=())
            for i in range(3)
        ]
        graph = build_narrative_graph(eps, adjacency_threshold=timedelta(0))
        G = graph.to_networkx()
        assert G.number_of_nodes() == graph.node_count()

    def test_networkx_edge_count(self, base_time):
        eps = [
            Episode(start_time=base_time + timedelta(hours=i),
                    end_time=base_time + timedelta(hours=i + 1), events=())
            for i in range(3)
        ]
        graph = build_narrative_graph(eps, adjacency_threshold=timedelta(0))
        G = graph.to_networkx()
        assert G.number_of_edges() == graph.edge_count()

    def test_networkx_node_attributes(self, base_time):
        ep = make_episode(base_time, 0, 2, ["positive", "negative"])
        graph = build_narrative_graph([ep])
        G = graph.to_networkx()

        attrs = G.nodes[0]
        assert "start_time" in attrs
        assert "end_time" in attrs
        assert "emotion_label" in attrs
        assert "event_count" in attrs
        assert attrs["event_count"] == 2

    def test_networkx_edge_attributes(self, base_time):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=2),
            events=(),
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=3),
            events=(),
        )
        graph = build_narrative_graph([ep1, ep2])
        G = graph.to_networkx()

        for u, v, data in G.edges(data=True):
            assert "relation" in data
            assert data["relation"] in ("overlapping", "adjacent", "disjoint")

    def test_networkx_edge_has_weight(self, base_time):
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=2),
            events=(),
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=3),
            events=(),
        )
        graph = build_narrative_graph([ep1, ep2])
        G = graph.to_networkx()

        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert 0.0 <= data["weight"] <= 1.0

    def test_empty_graph_to_networkx(self):
        graph = build_narrative_graph([])
        G = graph.to_networkx()
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_dominant_label_picks_majority(self, base_time):
        ep = make_episode(base_time, 0, 2, ["neutral", "positive", "positive"])
        graph = build_narrative_graph([ep])
        G = graph.to_networkx()
        assert G.nodes[0]["emotion_label"] == "positive"

    def test_empty_episode_label_is_none(self, base_time):
        ep = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        graph = build_narrative_graph([ep])
        G = graph.to_networkx()
        assert G.nodes[0]["emotion_label"] is None


# ---------------------------------------------------------------------------
# 6.  Edges in Response
# ---------------------------------------------------------------------------

class TestEdgesInResponse:

    def test_edges_key_present_on_empty_graph(self):
        graph = build_narrative_graph([])
        result = analyze_narrative_graph(graph)
        assert "edges" in result
        assert result["edges"] == []

    def test_edges_count_matches_summary(self, base_time):
        ep1 = make_episode(base_time, 0, 1, ["positive"])
        ep2 = make_episode(base_time, 1, 1, ["negative"])
        graph = build_narrative_graph([ep1, ep2], adjacency_threshold=timedelta(0))
        result = analyze_narrative_graph(graph)
        assert len(result["edges"]) == result["graph_summary"]["edge_count"]

    def test_edge_structure(self, base_time):
        ep1 = make_episode(base_time, 0, 1, ["positive"])
        ep2 = make_episode(base_time, 1, 1, ["negative"])
        graph = build_narrative_graph([ep1, ep2], adjacency_threshold=timedelta(0))
        result = analyze_narrative_graph(graph)
        edge = result["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "relation" in edge
        assert "weight" in edge
        assert edge["source"] < edge["target"]  # DAG ordering
        assert 0.0 <= edge["weight"] <= 1.0

    def test_disconnected_graph_has_no_edges(self, base_time):
        ep1 = make_episode(base_time, 0, 1, ["positive"])
        ep2 = make_episode(base_time, 48, 1, ["negative"])  # 48h apart
        graph = build_narrative_graph(
            [ep1, ep2], adjacency_threshold=timedelta(hours=1)
        )
        result = analyze_narrative_graph(graph)
        assert result["edges"] == []
