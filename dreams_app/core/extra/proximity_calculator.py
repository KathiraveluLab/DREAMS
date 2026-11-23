"""Multi-dimensional proximity calculator for location-based analysis.

Implements the proximity calculation framework defined in TEST_PLAN.md,
extending basic geographic distance with categorical, linguistic, and
cultural similarity dimensions.
"""

import math
from typing import Dict, List, Set, Tuple


# Categorical proximity mapping based on place type relationships
# Stored as sorted tuples to avoid redundancy
CATEGORY_RELATIONSHIPS = {
    ('clinic', 'hospital'): 0.5,
    ('church', 'temple'): 0.5,
    ('cafe', 'restaurant'): 0.5,
}


def categorical_proximity(type1: str, type2: str) -> float:
    """Calculate categorical proximity between two place types.
    
    Returns 1.0 for identical types, 0.5 for related types (e.g., hospital/clinic),
    and 0.0 for unrelated types.
    
    Args:
        type1: Place type of first location (e.g., 'church', 'hospital')
        type2: Place type of second location
        
    Returns:
        Proximity score between 0.0 and 1.0
        
    Examples:
        >>> categorical_proximity('church', 'church')
        1.0
        >>> categorical_proximity('hospital', 'clinic')
        0.5
        >>> categorical_proximity('church', 'restaurant')
        0.0
    """
    if not type1 or not type2:
        return 0.0
        
    type1_lower = type1.lower().strip()
    type2_lower = type2.lower().strip()
    
    if type1_lower == type2_lower:
        return 1.0
    
    # Normalize pair to sorted tuple for symmetric lookup
    pair = tuple(sorted([type1_lower, type2_lower]))
    return CATEGORY_RELATIONSHIPS.get(pair, 0.0)


def linguistic_similarity(language1: str, language2: str) -> float:
    """Calculate linguistic similarity between two languages.
    
    Returns 1.0 if languages match, 0.0 otherwise. Future versions could
    implement language family similarity (e.g., Spanish/Portuguese = 0.7).
    
    Args:
        language1: Language code or name (e.g., 'english', 'portuguese')
        language2: Language code or name
        
    Returns:
        Similarity score between 0.0 and 1.0
        
    Examples:
        >>> linguistic_similarity('portuguese', 'portuguese')
        1.0
        >>> linguistic_similarity('english', 'spanish')
        0.0
    """
    if not language1 or not language2:
        return 0.0
        
    return 1.0 if language1.lower().strip() == language2.lower().strip() else 0.0


def cultural_similarity(tags1: List[str], tags2: List[str]) -> float:
    """Calculate cultural similarity using Jaccard index on cultural tags.
    
    Jaccard index = |intersection| / |union|
    
    Args:
        tags1: List of cultural tags for first location (e.g., ['european', 'catholic'])
        tags2: List of cultural tags for second location
        
    Returns:
        Jaccard similarity score between 0.0 and 1.0
        
    Examples:
        >>> cultural_similarity(['european', 'catholic'], ['european', 'traditional'])
        0.333...  # 1 common / 3 total unique
        >>> cultural_similarity(['a', 'b'], ['a', 'b'])
        1.0
        >>> cultural_similarity(['a'], ['b'])
        0.0
    """
    if not tags1 and not tags2:
        return 1.0  # Both empty = identical
    if not tags1 or not tags2:
        return 0.0  # One empty = no similarity
        
    set1: Set[str] = {tag.lower().strip() for tag in tags1 if tag}
    set2: Set[str] = {tag.lower().strip() for tag in tags2 if tag}
    
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
        
    intersection = set1 & set2
    union = set1 | set2
    
    return len(intersection) / len(union) if union else 0.0


def composite_proximity(
    place1: Dict,
    place2: Dict,
    weights: Dict[str, float] = None
) -> float:
    """Calculate composite proximity using weighted sum of all dimensions.
    
    Combines geographic, categorical, linguistic, and cultural proximity
    into a single score.
    
    Args:
        place1: First location dict with keys: 'type', 'language', 'cultural_tags', 'geo_proximity'
        place2: Second location dict with same structure. If both places have a
                'geo_proximity', their mean is used, so the result does not
                depend on the order of the arguments.
        weights: Optional weight dict with keys: 'geo', 'cat', 'ling', 'cult'
                 Defaults to α=0.3, β=0.4, γ=0.15, δ=0.15 from TEST_PLAN
                 Missing keys will use default values.
                 
    Returns:
        Composite proximity score between 0.0 and 1.0

    Missing data: a dimension takes part only if both places have a value for it
    ('geo_proximity' on either place counts). Missing dimensions are left out and
    the weights are renormalised over the rest. If no dimension is known, the
    score is 0.0.
        
    Examples:
        >>> p1 = {'type': 'church', 'language': 'english', 'cultural_tags': ['christian']}
        >>> p2 = {'type': 'church', 'language': 'english', 'cultural_tags': ['christian']}
        >>> composite_proximity(p1, p2)  # Perfect match
        1.0
    """
    default_weights = {'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15}
    weights = {**default_weights, **(weights or {})}
    
    # Only the dimensions both places have data for take part (TEST_PLAN PC-EC-003,
    # "use available dimensions"). A missing value is unknown, not similar or different.
    scores: Dict[str, float] = {}

    # Geographic proximity from the pre-calculated values. Use both places so the
    # score does not depend on which one is passed first.
    geo_values = [
        p['geo_proximity'] for p in (place1, place2)
        if p.get('geo_proximity') is not None
    ]
    if geo_values:
        scores['geo'] = sum(geo_values) / len(geo_values)

    if place1.get('type') and place2.get('type'):
        scores['cat'] = categorical_proximity(place1['type'], place2['type'])

    if place1.get('language') and place2.get('language'):
        scores['ling'] = linguistic_similarity(place1['language'], place2['language'])

    if place1.get('cultural_tags') and place2.get('cultural_tags'):
        scores['cult'] = cultural_similarity(place1['cultural_tags'], place2['cultural_tags'])

    # Weighted sum, renormalised over the dimensions that are present
    total_weight = sum(weights[name] for name in scores)
    if total_weight == 0:
        return 0.0

    return sum(weights[name] * score for name, score in scores.items()) / total_weight


def normalize_geographic_distance(distance_km: float, max_distance_km: float = 500.0) -> float:
    """Normalize geographic distance to [0, 1] proximity score.
    
    Uses exponential decay: proximity = exp(-distance / scale)
    where scale = max_distance_km / 3 (so max_distance → ~0.05 proximity)
    
    Args:
        distance_km: Distance in kilometers
        max_distance_km: Maximum meaningful distance (default 500 km)
        
    Returns:
        Proximity score between 0.0 and 1.0 (1.0 = same location)
        
    Examples:
        >>> normalize_geographic_distance(0.0)
        1.0
        >>> normalize_geographic_distance(500.0)  # doctest: +ELLIPSIS
        0.04...
    """
    if distance_km < 0:
        raise ValueError("Distance cannot be negative")
        
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be greater than zero")
    scale = max_distance_km / 3.0
    return math.exp(-distance_km / scale)


def cluster_places_dbscan(
    places: List[Dict],
    eps: float = 0.4,
    min_samples: int = 2,
    weights: Dict[str, float] = None,
) -> List[int]:
    """Cluster places using density-based clustering (DBSCAN) over composite proximity.

    Constructs a precomputed pairwise distance matrix D_ij = 1.0 - P_composite(L_i, L_j)
    and groups locations into socio-cultural micro-environments without requiring a
    pre-specified cluster count. Noise points receive label -1.

    Args:
        places: List of place dicts accepted by composite_proximity.
        eps: Maximum composite distance (1 - proximity) between neighbor places.
        min_samples: Minimum number of places required to form a dense region.
        weights: Optional dimension weights passed to composite_proximity.

    Returns:
        List of integer cluster labels (one per place in input order; -1 for noise).
    """
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    n = len(places)
    if n == 0:
        return []

    dist_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            prox = composite_proximity(places[i], places[j], weights=weights)
            dist = max(0.0, 1.0 - prox)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    try:
        import numpy as np
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(np.asarray(dist_matrix, dtype=float))
        return [int(lbl) for lbl in labels]
    except ImportError:
        neighbors = [
            [j for j in range(n) if dist_matrix[i][j] <= eps]
            for i in range(n)
        ]
        core_points = {i for i in range(n) if len(neighbors[i]) >= min_samples}
        labels = [-1] * n
        cluster_id = 0
        visited: Set[int] = set()

        for i in range(n):
            if i in visited or i not in core_points:
                continue
            visited.add(i)
            labels[i] = cluster_id
            queue = list(neighbors[i])
            idx = 0
            while idx < len(queue):
                q = queue[idx]
                idx += 1
                if q not in visited:
                    visited.add(q)
                    if q in core_points:
                        for nb in neighbors[q]:
                            if nb not in queue:
                                queue.append(nb)
                if labels[q] == -1:
                    labels[q] = cluster_id
            cluster_id += 1
        return labels

