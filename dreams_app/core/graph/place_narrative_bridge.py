from typing import Dict, List, Tuple

from dreams_app.core.extra.place_emotion_signature import (
    PlaceEmotionSignature,
    chime_proximity,
)
from dreams_app.core.graph.temporal_narrative_graph import (
    NarrativeEdge,
    TemporalNarrativeGraph,
)


def enrich_narrative_edges_with_place_proximity(
    graph: TemporalNarrativeGraph,
    episode_place_signatures: Dict[int, PlaceEmotionSignature],
    place_weight: float = 0.3
) -> TemporalNarrativeGraph:
    """
    Blend temporal edge weights with CHIME-based place proximity.

    For each edge where both episodes have a place signature, the new
    weight is:
        (1 - place_weight) * temporal_weight + place_weight * chime_proximity

    Episodes with no signature keep their original weight unchanged.

    place_weight=0.0 → pure temporal (no change)
    place_weight=1.0 → pure place proximity

    Raises ValueError if place_weight is outside [0.0, 1.0].
    """
    if not (0.0 <= place_weight <= 1.0):
        raise ValueError(f"place_weight must be in [0.0, 1.0], got {place_weight}")

    enriched_edges = []
    for edge in graph.edges:
        sig_a = episode_place_signatures.get(edge.source_index)
        sig_b = episode_place_signatures.get(edge.target_index)

        if sig_a is not None and sig_b is not None:
            place_prox = chime_proximity(sig_a, sig_b)
            new_weight = (1.0 - place_weight) * edge.weight + place_weight * place_prox
            new_weight = max(0.0, min(1.0, new_weight))
        else:
            new_weight = edge.weight

        enriched_edges.append(NarrativeEdge(
            source_index=edge.source_index,
            target_index=edge.target_index,
            relation=edge.relation,
            weight=new_weight
        ))

    return TemporalNarrativeGraph(
        nodes=graph.nodes,
        edges=tuple(enriched_edges),
        adjacency_threshold=graph.adjacency_threshold
    )


def compute_place_proximity_matrix(
    signatures: Dict[str, PlaceEmotionSignature]
) -> Dict[str, Dict[str, float]]:
    """
    Pairwise CHIME proximity between all place types.

    Returns a symmetric nested dict where matrix[a][b] == matrix[b][a]
    and matrix[a][a] == 1.0.
    """
    place_types = list(signatures.keys())
    matrix: Dict[str, Dict[str, float]] = {}

    for pt_a in place_types:
        matrix[pt_a] = {}
        for pt_b in place_types:
            if pt_a == pt_b:
                matrix[pt_a][pt_b] = 1.0
            else:
                matrix[pt_a][pt_b] = chime_proximity(signatures[pt_a], signatures[pt_b])

    return matrix


def find_emotionally_proximate_pairs(
    signatures: Dict[str, PlaceEmotionSignature],
    threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    Place-type pairs whose CHIME proximity is at or above threshold.

    Useful for surfacing unexpected connections - places that are
    categorically different but emotionally similar for this person.

    Returns list of (place_type_a, place_type_b, score) sorted descending.
    No self-pairs, no duplicates.
    """
    place_types = list(signatures.keys())
    pairs = []

    for i, pt_a in enumerate(place_types):
        for pt_b in place_types[i + 1:]:
            score = chime_proximity(signatures[pt_a], signatures[pt_b])
            if score >= threshold:
                pairs.append((pt_a, pt_b, score))

    return sorted(pairs, key=lambda x: x[2], reverse=True)
