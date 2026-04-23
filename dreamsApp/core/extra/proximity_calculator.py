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
    ('church', 'church'): 1.0,
    ('hospital', 'hospital'): 1.0,
    ('restaurant', 'restaurant'): 1.0,
    ('park', 'park'): 1.0,
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
        place2: Second location dict with same structure
        weights: Optional weight dict with keys: 'geo', 'cat', 'ling', 'cult'
                 Defaults to α=0.3, β=0.4, γ=0.15, δ=0.15 from TEST_PLAN
                 Missing keys will use default values.
                 
    Returns:
        Composite proximity score between 0.0 and 1.0
        
    Examples:
        >>> p1 = {'type': 'church', 'language': 'english', 'cultural_tags': ['christian']}
        >>> p2 = {'type': 'church', 'language': 'english', 'cultural_tags': ['christian']}
        >>> composite_proximity(p1, p2)  # Perfect match
        1.0
    """
    default_weights = {'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15}
    weights = {**default_weights, **(weights or {})}
    
    # Geographic proximity from pre-calculated value
    geo_prox = place1.get('geo_proximity', 1.0)
    
    # Categorical proximity
    cat_prox = categorical_proximity(
        place1.get('type', ''),
        place2.get('type', '')
    )
    
    # Linguistic similarity
    ling_prox = linguistic_similarity(
        place1.get('language', ''),
        place2.get('language', '')
    )
    
    # Cultural similarity
    cult_prox = cultural_similarity(
        place1.get('cultural_tags', []),
        place2.get('cultural_tags', [])
    )
    
    # Weighted sum
    # Weighted sum
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    composite = (
        weights['geo'] * geo_prox +
        weights['cat'] * cat_prox +
        weights['ling'] * ling_prox +
        weights['cult'] * cult_prox
    ) / total_weight
    
    return composite


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
