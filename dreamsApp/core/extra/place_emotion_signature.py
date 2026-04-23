from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


CHIME_DIMENSIONS = ['Connectedness', 'Hope', 'Identity', 'Meaning', 'Empowerment']


@dataclass
class PlaceEmotionSignature:
    """
    Mean CHIME profile for a location type across all visits.

    place_type  : category label, e.g. 'church', 'park'
    chime_vector: mean score per CHIME dimension
    visit_count : number of visits used to build the profile
    volatility  : RMS std-dev across dimensions (low = emotionally stable place)
    """
    place_type: str
    chime_vector: Dict[str, float]
    visit_count: int = 0
    volatility: float = 0.0


def build_place_signature(
    place_type: str,
    chime_results: List[Dict[str, float]]
) -> PlaceEmotionSignature:
    """
    Aggregate per-visit CHIME dicts into a single PlaceEmotionSignature.

    Missing dimensions in any visit default to 0.0.
    Volatility is the RMS of per-dimension population std-devs.
    """
    if not chime_results:
        return PlaceEmotionSignature(
            place_type=place_type,
            chime_vector={dim: 0.0 for dim in CHIME_DIMENSIONS},
            visit_count=0,
            volatility=0.0
        )

    n = len(chime_results)

    mean_vector = {
        dim: sum(r.get(dim, 0.0) for r in chime_results) / n
        for dim in CHIME_DIMENSIONS
    }

    total_variance = sum(
        sum((r.get(dim, 0.0) - mean_vector[dim]) ** 2 for r in chime_results) / n
        for dim in CHIME_DIMENSIONS
    )
    volatility = math.sqrt(total_variance / len(CHIME_DIMENSIONS))

    return PlaceEmotionSignature(
        place_type=place_type,
        chime_vector=mean_vector,
        visit_count=n,
        volatility=volatility
    )


def chime_proximity(
    sig_a: PlaceEmotionSignature,
    sig_b: PlaceEmotionSignature,
    min_visits: int = 1
) -> float:
    """
    Cosine similarity between two place-type CHIME vectors.

    Returns 0.5 when either signature has fewer than min_visits
    (not enough data to be confident either way).
    """
    if sig_a.visit_count < min_visits or sig_b.visit_count < min_visits:
        return 0.5

    vec_a = [sig_a.chime_vector.get(d, 0.0) for d in CHIME_DIMENSIONS]
    vec_b = [sig_b.chime_vector.get(d, 0.0) for d in CHIME_DIMENSIONS]

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def detect_recovery_anchors(
    signatures: Dict[str, PlaceEmotionSignature],
    anchor_dimensions: List[str] = None,
    threshold: float = 0.6,
    min_visits: int = 2
) -> List[Tuple[str, float, str]]:
    """
    Return place types that consistently trigger recovery-relevant CHIME dimensions.

    anchor_score = best_dimension_score * (1 - volatility)
    High score + low volatility = stable recovery anchor.

    Returns list of (place_type, anchor_score, dominant_dimension)
    sorted by anchor_score descending.
    """
    if anchor_dimensions is None:
        anchor_dimensions = ['Hope', 'Connectedness']

    anchors = []
    for place_type, sig in signatures.items():
        if sig.visit_count < min_visits:
            continue

        best_dim = max(anchor_dimensions, key=lambda d: sig.chime_vector.get(d, 0.0))
        best_score = sig.chime_vector.get(best_dim, 0.0)
        anchor_score = best_score * max(0.0, 1.0 - sig.volatility)

        if anchor_score >= threshold:
            anchors.append((place_type, anchor_score, best_dim))

    return sorted(anchors, key=lambda x: x[1], reverse=True)


def get_dominant_chime_dimension(sig: PlaceEmotionSignature) -> Tuple[str, float]:
    """Return (dimension_name, score) for the highest-scoring CHIME dimension."""
    if not sig.chime_vector:
        return ('None', 0.0)
    best_dim = max(sig.chime_vector, key=sig.chime_vector.get)
    return (best_dim, sig.chime_vector[best_dim])
