"""
Proximity Trajectory Analysis

Tracks how a person's emotional response to each place type evolves
over time, and detects whether visits to recovery anchor places
correlate with positive emotional shifts.

This answers the core research question:
"How do emotions attached to a class of places (e.g., any churches)
evolve across a recovery journey?"

Builds on:
- place_emotion_signature.py: CHIME profiles per place type
- temporal_narrative_graph.py: emotional episode structure
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math


CHIME_DIMENSIONS = ['Connectedness', 'Hope', 'Identity', 'Meaning', 'Empowerment']


@dataclass
class PlaceVisit:
    """
    A single visit to a place type with its emotional context.

    place_type : category of place (e.g., 'church', 'park')
    timestamp  : when the visit occurred
    chime      : CHIME scores at time of visit
    """
    place_type: str
    timestamp: datetime
    chime: Dict[str, float]


@dataclass
class PlaceTypeTrajectory:
    """
    Emotional trajectory for a single place type over time.

    place_type   : category of place
    visits       : chronologically ordered list of PlaceVisit
    trend        : per-dimension slope (positive = improving over time)
    volatility   : mean std-dev across visits (emotional consistency)
    visit_count  : total number of visits
    """
    place_type: str
    visits: List[PlaceVisit]
    trend: Dict[str, float]
    volatility: float
    visit_count: int


def build_place_trajectories(
    visits: List[PlaceVisit],
) -> Dict[str, PlaceTypeTrajectory]:
    """
    Group visits by place type and compute emotional trajectory per type.

    For each place type, computes:
    - trend: linear slope of each CHIME dimension over visits
      (positive = dimension is increasing across visits = improving)
    - volatility: mean RMS std-dev across CHIME dimensions

    Args:
        visits: List of PlaceVisit objects, any order

    Returns:
        Dict of place_type -> PlaceTypeTrajectory, sorted by visit time
    """
    import math

    # Group by place type, sort by time
    grouped: Dict[str, List[PlaceVisit]] = {}
    for v in visits:
        grouped.setdefault(v.place_type, []).append(v)
    for pt in grouped:
        grouped[pt].sort(key=lambda v: v.timestamp)

    trajectories = {}
    for place_type, place_visits in grouped.items():
        n = len(place_visits)

        # Compute per-dimension linear slope using least squares
        # slope > 0 means dimension is increasing over visits (recovery signal)
        trend = {}
        for dim in CHIME_DIMENSIONS:
            scores = [v.chime.get(dim, 0.0) for v in place_visits]
            if n < 2:
                trend[dim] = 0.0
            else:
                # x = visit index (0, 1, 2, ...), y = score
                x_mean = (n - 1) / 2.0
                y_mean = sum(scores) / n
                numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                trend[dim] = numerator / denominator if denominator != 0 else 0.0

        # Volatility: mean RMS std-dev across dimensions
        total_variance = 0.0
        for dim in CHIME_DIMENSIONS:
            scores = [v.chime.get(dim, 0.0) for v in place_visits]
            mean = sum(scores) / n
            variance = sum((s - mean) ** 2 for s in scores) / n
            total_variance += variance
        volatility = math.sqrt(total_variance / len(CHIME_DIMENSIONS))

        trajectories[place_type] = PlaceTypeTrajectory(
            place_type=place_type,
            visits=place_visits,
            trend=trend,
            volatility=volatility,
            visit_count=n,
        )

    return trajectories


def detect_recovery_correlations(
    trajectories: Dict[str, PlaceTypeTrajectory],
    recovery_dimensions: List[str] = None,
    min_visits: int = 2,
    trend_threshold: float = 0.01,
) -> List[Tuple[str, str, float]]:
    """
    Detect place types where recovery-relevant CHIME dimensions are improving.

    A positive trend in Hope or Connectedness across visits to a place type
    suggests that place is contributing to recovery for this person.

    Args:
        trajectories: Output of build_place_trajectories()
        recovery_dimensions: CHIME dimensions to check (default: Hope, Connectedness)
        min_visits: Minimum visits required for reliable trend
        trend_threshold: Minimum slope to count as meaningful improvement

    Returns:
        List of (place_type, dimension, slope) tuples where slope > threshold,
        sorted by slope descending
    """
    if recovery_dimensions is None:
        recovery_dimensions = ['Hope', 'Connectedness']

    correlations = []
    for place_type, traj in trajectories.items():
        if traj.visit_count < min_visits:
            continue
        for dim in recovery_dimensions:
            slope = traj.trend.get(dim, 0.0)
            if slope > trend_threshold:
                correlations.append((place_type, dim, slope))

    return sorted(correlations, key=lambda x: x[2], reverse=True)


def get_dominant_trend_dimension(traj: PlaceTypeTrajectory) -> Tuple[str, float]:
    """
    Return the CHIME dimension with the strongest positive trend for a place type.

    Args:
        traj: PlaceTypeTrajectory

    Returns:
        (dimension_name, slope) for the dimension improving most
    """
    if not traj.trend:
        return ('None', 0.0)
    best_dim = max(traj.trend, key=traj.trend.get)
    return (best_dim, traj.trend[best_dim])


def summarize_trajectories(
    trajectories: Dict[str, PlaceTypeTrajectory],
) -> Dict[str, Dict]:
    """
    Produce a human-readable summary of all place type trajectories.

    For each place type, returns:
    - visit_count
    - dominant_trend: dimension improving most
    - trend_direction: 'improving', 'stable', or 'declining'
    - volatility

    Args:
        trajectories: Output of build_place_trajectories()

    Returns:
        Dict of place_type -> summary dict
    """
    summary = {}
    for place_type, traj in trajectories.items():
        best_dim, best_slope = get_dominant_trend_dimension(traj)

        if best_slope > 0.02:
            direction = 'improving'
        elif best_slope < -0.02:
            direction = 'declining'
        else:
            direction = 'stable'

        summary[place_type] = {
            'visit_count': traj.visit_count,
            'dominant_trend': best_dim,
            'trend_slope': round(best_slope, 4),
            'trend_direction': direction,
            'volatility': round(traj.volatility, 4),
        }

    return summary
