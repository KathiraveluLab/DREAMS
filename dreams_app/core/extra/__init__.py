"""Extra analysis modules for DREAMS."""

from .proximity_calculator import (
    categorical_proximity,
    linguistic_similarity,
    cultural_similarity,
    composite_proximity,
    normalize_geographic_distance,
)

__all__ = [
    'categorical_proximity',
    'linguistic_similarity',
    'cultural_similarity',
    'composite_proximity',
    'normalize_geographic_distance',
]
