import pytest
from datetime import datetime, timedelta

from dreamsApp.core.extra.place_emotion_signature import (
    PlaceEmotionSignature,
    build_place_signature,
)
from dreamsApp.core.graph.place_narrative_bridge import (
    enrich_narrative_edges_with_place_proximity,
    compute_place_proximity_matrix,
    find_emotionally_proximate_pairs,
)
from dreamsApp.core.graph.temporal_narrative_graph import (
    NarrativeEdge,
    TemporalNarrativeGraph,
    build_narrative_graph,
)
from dreamsApp.core.graph.emotion_episode import Episode
from dreamsApp.core.graph.emotion_timeline import EmotionEvent


def make_episode(start_offset_days: int, duration_days: int = 1) -> Episode:
    base = datetime(2024, 1, 1)
    start = base + timedelta(days=start_offset_days)
    end = start + timedelta(days=duration_days)
    return Episode(
        start_time=start,
        end_time=end,
        events=(EmotionEvent(timestamp=start, emotion_label='positive', score=0.8),)
    )


def make_simple_graph() -> TemporalNarrativeGraph:
    return build_narrative_graph(
        [make_episode(0), make_episode(2), make_episode(5)],
        adjacency_threshold=timedelta(days=3),
        include_disjoint_edges=True
    )


def make_sig(place_type: str, hope: float, connectedness: float, visit_count: int = 3) -> PlaceEmotionSignature:
    return PlaceEmotionSignature(
        place_type=place_type,
        chime_vector={'Hope': hope, 'Connectedness': connectedness, 'Identity': 0.3, 'Meaning': 0.5, 'Empowerment': 0.4},
        visit_count=visit_count,
        volatility=0.1
    )


class TestEnrichNarrativeEdges:

    def test_returns_temporal_narrative_graph(self):
        assert isinstance(enrich_narrative_edges_with_place_proximity(make_simple_graph(), {}), TemporalNarrativeGraph)

    def test_same_node_count(self):
        graph = make_simple_graph()
        assert enrich_narrative_edges_with_place_proximity(graph, {}).node_count() == graph.node_count()

    def test_same_edge_count(self):
        graph = make_simple_graph()
        assert enrich_narrative_edges_with_place_proximity(graph, {}).edge_count() == graph.edge_count()

    def test_no_signatures_preserves_weights(self):
        graph = make_simple_graph()
        result = enrich_narrative_edges_with_place_proximity(graph, {})
        for orig, enriched in zip(graph.edges, result.edges):
            assert orig.weight == enriched.weight

    def test_similar_place_types_increase_weight(self):
        graph = make_simple_graph()
        sigs = {0: make_sig('church', hope=0.9, connectedness=0.8), 1: make_sig('community_center', hope=0.85, connectedness=0.82)}
        result = enrich_narrative_edges_with_place_proximity(graph, sigs, place_weight=0.3)
        orig = next(e for e in graph.edges if e.source_index == 0 and e.target_index == 1)
        new = next(e for e in result.edges if e.source_index == 0 and e.target_index == 1)
        assert new.weight >= orig.weight * 0.9

    def test_dissimilar_place_types_may_lower_weight(self):
        graph = make_simple_graph()
        sigs = {
            0: make_sig('church', hope=0.9, connectedness=0.8),
            1: PlaceEmotionSignature('hospital', {'Hope': 0.2, 'Connectedness': 0.2, 'Identity': 0.8, 'Meaning': 0.4, 'Empowerment': 0.3}, visit_count=3, volatility=0.1),
        }
        result = enrich_narrative_edges_with_place_proximity(graph, sigs, place_weight=0.5)
        orig = next(e for e in graph.edges if e.source_index == 0 and e.target_index == 1)
        new = next(e for e in result.edges if e.source_index == 0 and e.target_index == 1)
        assert new.weight <= orig.weight + 0.1

    def test_enriched_weights_in_valid_range(self):
        graph = make_simple_graph()
        sigs = {0: make_sig('church', 0.9, 0.8), 1: make_sig('park', 0.7, 0.75), 2: make_sig('hospital', 0.2, 0.2)}
        result = enrich_narrative_edges_with_place_proximity(graph, sigs)
        for edge in result.edges:
            assert 0.0 <= edge.weight <= 1.0

    def test_relations_preserved(self):
        graph = make_simple_graph()
        result = enrich_narrative_edges_with_place_proximity(graph, {0: make_sig('church', 0.9, 0.8)})
        for orig, enriched in zip(graph.edges, result.edges):
            assert orig.relation == enriched.relation

    def test_invalid_place_weight_raises(self):
        graph = make_simple_graph()
        with pytest.raises(ValueError):
            enrich_narrative_edges_with_place_proximity(graph, {}, place_weight=1.5)
        with pytest.raises(ValueError):
            enrich_narrative_edges_with_place_proximity(graph, {}, place_weight=-0.1)

    def test_empty_graph_returns_empty(self):
        empty = TemporalNarrativeGraph(nodes=(), edges=())
        result = enrich_narrative_edges_with_place_proximity(empty, {})
        assert result.edge_count() == 0

    def test_place_weight_zero_preserves_all_weights(self):
        graph = make_simple_graph()
        sigs = {0: make_sig('church', 0.9, 0.8), 1: make_sig('park', 0.7, 0.75)}
        result = enrich_narrative_edges_with_place_proximity(graph, sigs, place_weight=0.0)
        for orig, enriched in zip(graph.edges, result.edges):
            assert abs(orig.weight - enriched.weight) < 1e-9

    def test_partial_signatures_only_enriches_covered_edges(self):
        graph = make_simple_graph()
        sigs = {0: make_sig('church', 0.9, 0.8)}
        result = enrich_narrative_edges_with_place_proximity(graph, sigs, place_weight=0.5)
        orig_01 = next(e for e in graph.edges if e.source_index == 0 and e.target_index == 1)
        new_01 = next(e for e in result.edges if e.source_index == 0 and e.target_index == 1)
        assert orig_01.weight == new_01.weight


class TestComputePlaceProximityMatrix:

    @pytest.fixture
    def signatures(self):
        return {
            'church': make_sig('church', hope=0.9, connectedness=0.8),
            'community_center': make_sig('community_center', hope=0.85, connectedness=0.82),
            'hospital': PlaceEmotionSignature('hospital', {'Hope': 0.2, 'Connectedness': 0.2, 'Identity': 0.8, 'Meaning': 0.4, 'Empowerment': 0.3}, visit_count=3, volatility=0.1),
        }

    def test_returns_dict(self, signatures):
        assert isinstance(compute_place_proximity_matrix(signatures), dict)

    def test_diagonal_is_one(self, signatures):
        matrix = compute_place_proximity_matrix(signatures)
        for pt in signatures:
            assert abs(matrix[pt][pt] - 1.0) < 1e-9

    def test_symmetric(self, signatures):
        matrix = compute_place_proximity_matrix(signatures)
        for pt_a in signatures:
            for pt_b in signatures:
                assert abs(matrix[pt_a][pt_b] - matrix[pt_b][pt_a]) < 1e-9

    def test_all_pairs_present(self, signatures):
        matrix = compute_place_proximity_matrix(signatures)
        for pt_a in signatures:
            for pt_b in signatures:
                assert pt_b in matrix[pt_a]

    def test_church_community_higher_than_church_hospital(self, signatures):
        matrix = compute_place_proximity_matrix(signatures)
        assert matrix['church']['community_center'] > matrix['church']['hospital']

    def test_empty_returns_empty(self):
        assert compute_place_proximity_matrix({}) == {}

    def test_values_in_range(self, signatures):
        matrix = compute_place_proximity_matrix(signatures)
        for pt_a in signatures:
            for pt_b in signatures:
                assert 0.0 <= matrix[pt_a][pt_b] <= 1.0


class TestFindEmotionallyProximatePairs:

    @pytest.fixture
    def signatures(self):
        return {
            'church': make_sig('church', hope=0.9, connectedness=0.8),
            'community_center': make_sig('community_center', hope=0.85, connectedness=0.82),
            'hospital': PlaceEmotionSignature('hospital', {'Hope': 0.2, 'Connectedness': 0.2, 'Identity': 0.8, 'Meaning': 0.4, 'Empowerment': 0.3}, visit_count=3, volatility=0.1),
        }

    def test_returns_list(self, signatures):
        assert isinstance(find_emotionally_proximate_pairs(signatures), list)

    def test_tuple_structure(self, signatures):
        for pt_a, pt_b, score in find_emotionally_proximate_pairs(signatures, threshold=0.0):
            assert isinstance(pt_a, str)
            assert isinstance(pt_b, str)
            assert isinstance(score, float)

    def test_no_self_pairs(self, signatures):
        for pt_a, pt_b, _ in find_emotionally_proximate_pairs(signatures, threshold=0.0):
            assert pt_a != pt_b

    def test_sorted_descending(self, signatures):
        result = find_emotionally_proximate_pairs(signatures, threshold=0.0)
        scores = [r[2] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filters(self, signatures):
        all_pairs = find_emotionally_proximate_pairs(signatures, threshold=0.0)
        filtered = find_emotionally_proximate_pairs(signatures, threshold=0.9)
        assert len(filtered) <= len(all_pairs)
        for _, _, score in filtered:
            assert score >= 0.9

    def test_church_community_is_top_pair(self, signatures):
        result = find_emotionally_proximate_pairs(signatures, threshold=0.0)
        if result:
            assert set([result[0][0], result[0][1]]) == {'church', 'community_center'}

    def test_no_duplicate_pairs(self, signatures):
        result = find_emotionally_proximate_pairs(signatures, threshold=0.0)
        seen = set()
        for pt_a, pt_b, _ in result:
            pair = frozenset([pt_a, pt_b])
            assert pair not in seen
            seen.add(pair)

    def test_empty_returns_empty(self):
        assert find_emotionally_proximate_pairs({}) == []

    def test_single_signature_returns_empty(self):
        assert find_emotionally_proximate_pairs({'church': make_sig('church', 0.9, 0.8)}, threshold=0.0) == []
