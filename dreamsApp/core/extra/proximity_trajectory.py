import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math


CHIME_DIMENSIONS = ['Connectedness', 'Hope', 'Identity', 'Meaning', 'Empowerment']

# Minimum slope magnitude to classify a trend as improving or declining.
# Aligned with the default trend_threshold in detect_recovery_correlations.
TREND_THRESHOLD = 0.01


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
        Dict of place_type -> PlaceTypeTrajectory, grouped in chronological
        order of first visit per place type
    """
    sorted_visits = sorted(visits, key=lambda v: v.timestamp)

    grouped: Dict[str, List[PlaceVisit]] = {}
    for v in sorted_visits:
        grouped.setdefault(v.place_type, []).append(v)

    trajectories = {}
    for place_type, place_visits in grouped.items():
        n = len(place_visits)

        x_mean = (n - 1) / 2.0
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        trend: Dict[str, float] = {}
        total_variance = 0.0

        for dim in CHIME_DIMENSIONS:
            scores = [v.chime.get(dim, 0.0) for v in place_visits]
            y_mean = sum(scores) / n

            if denominator == 0:
                trend[dim] = 0.0
            else:
                numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
                trend[dim] = numerator / denominator

            total_variance += sum((s - y_mean) ** 2 for s in scores) / n

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
    recovery_dimensions: Optional[List[str]] = None,
    min_visits: int = 2,
    trend_threshold: float = TREND_THRESHOLD,
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
    - trend_slope: slope of dominant dimension
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

        if best_slope > TREND_THRESHOLD:
            direction = 'improving'
        elif best_slope < -TREND_THRESHOLD:
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
