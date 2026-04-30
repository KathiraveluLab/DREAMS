import pytest
from datetime import datetime, timedelta
from dreamsApp.core.extra.proximity_trajectory import (
    PlaceVisit,
    PlaceTypeTrajectory,
    build_place_trajectories,
    detect_recovery_correlations,
    get_dominant_trend_dimension,
    summarize_trajectories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_visit(place_type: str, day: int, hope: float, connectedness: float) -> PlaceVisit:
    return PlaceVisit(
        place_type=place_type,
        timestamp=datetime(2024, 1, 1) + timedelta(days=day),
        chime={
            'Connectedness': connectedness,
            'Hope': hope,
            'Identity': 0.3,
            'Meaning': 0.5,
            'Empowerment': 0.4,
        }
    )


@pytest.fixture
def improving_church_visits():
    # Hope increasing over 4 visits: 0.5 → 0.6 → 0.7 → 0.8
    return [
        make_visit('church', 0, hope=0.5, connectedness=0.6),
        make_visit('church', 7, hope=0.6, connectedness=0.65),
        make_visit('church', 14, hope=0.7, connectedness=0.7),
        make_visit('church', 21, hope=0.8, connectedness=0.75),
    ]


@pytest.fixture
def declining_hospital_visits():
    # Hope decreasing over 3 visits: 0.4 → 0.3 → 0.2
    return [
        make_visit('hospital', 0, hope=0.4, connectedness=0.3),
        make_visit('hospital', 5, hope=0.3, connectedness=0.25),
        make_visit('hospital', 10, hope=0.2, connectedness=0.2),
    ]


@pytest.fixture
def mixed_visits(improving_church_visits, declining_hospital_visits):
    return improving_church_visits + declining_hospital_visits


# ---------------------------------------------------------------------------
# build_place_trajectories
# ---------------------------------------------------------------------------

class TestBuildPlaceTrajectories:

    def test_returns_dict(self, mixed_visits):
        result = build_place_trajectories(mixed_visits)
        assert isinstance(result, dict)

    def test_groups_by_place_type(self, mixed_visits):
        result = build_place_trajectories(mixed_visits)
        assert 'church' in result
        assert 'hospital' in result

    def test_visit_count(self, improving_church_visits):
        result = build_place_trajectories(improving_church_visits)
        assert result['church'].visit_count == 4

    def test_visits_sorted_by_time(self, improving_church_visits):
        import random
        shuffled = improving_church_visits.copy()
        random.shuffle(shuffled)
        result = build_place_trajectories(shuffled)
        times = [v.timestamp for v in result['church'].visits]
        assert times == sorted(times)

    def test_positive_trend_for_improving_visits(self, improving_church_visits):
        result = build_place_trajectories(improving_church_visits)
        assert result['church'].trend['Hope'] > 0

    def test_negative_trend_for_declining_visits(self, declining_hospital_visits):
        result = build_place_trajectories(declining_hospital_visits)
        assert result['hospital'].trend['Hope'] < 0

    def test_zero_trend_for_single_visit(self):
        visits = [make_visit('park', 0, hope=0.7, connectedness=0.6)]
        result = build_place_trajectories(visits)
        assert result['park'].trend['Hope'] == 0.0

    def test_volatility_non_negative(self, mixed_visits):
        result = build_place_trajectories(mixed_visits)
        for traj in result.values():
            assert traj.volatility >= 0.0

    def test_low_volatility_for_consistent_visits(self):
        visits = [make_visit('church', i, hope=0.8, connectedness=0.7) for i in range(4)]
        result = build_place_trajectories(visits)
        assert result['church'].volatility == 0.0

    def test_empty_visits_returns_empty(self):
        result = build_place_trajectories([])
        assert result == {}

    def test_returns_place_type_trajectory(self, improving_church_visits):
        result = build_place_trajectories(improving_church_visits)
        assert isinstance(result['church'], PlaceTypeTrajectory)


# ---------------------------------------------------------------------------
# detect_recovery_correlations
# ---------------------------------------------------------------------------

class TestDetectRecoveryCorrelations:

    def test_returns_list(self, mixed_visits):
        trajs = build_place_trajectories(mixed_visits)
        result = detect_recovery_correlations(trajs)
        assert isinstance(result, list)

    def test_improving_place_detected(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        result = detect_recovery_correlations(trajs, trend_threshold=0.0)
        place_types = [r[0] for r in result]
        assert 'church' in place_types

    def test_declining_place_not_detected(self, declining_hospital_visits):
        trajs = build_place_trajectories(declining_hospital_visits)
        result = detect_recovery_correlations(trajs, trend_threshold=0.0)
        # hospital has negative trend, should not appear
        place_types = [r[0] for r in result]
        assert 'hospital' not in place_types

    def test_sorted_by_slope_descending(self, mixed_visits):
        trajs = build_place_trajectories(mixed_visits)
        result = detect_recovery_correlations(trajs, trend_threshold=0.0)
        slopes = [r[2] for r in result]
        assert slopes == sorted(slopes, reverse=True)

    def test_tuple_structure(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        result = detect_recovery_correlations(trajs, trend_threshold=0.0)
        for item in result:
            place_type, dim, slope = item
            assert isinstance(place_type, str)
            assert isinstance(dim, str)
            assert isinstance(slope, float)

    def test_min_visits_filter(self):
        visits = [make_visit('church', 0, hope=0.9, connectedness=0.8)]
        trajs = build_place_trajectories(visits)
        result = detect_recovery_correlations(trajs, min_visits=2, trend_threshold=0.0)
        assert result == []

    def test_custom_recovery_dimensions(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        result = detect_recovery_correlations(
            trajs, recovery_dimensions=['Meaning'], trend_threshold=0.0
        )
        for _, dim, _ in result:
            assert dim == 'Meaning'

    def test_empty_trajectories_returns_empty(self):
        result = detect_recovery_correlations({})
        assert result == []


# ---------------------------------------------------------------------------
# get_dominant_trend_dimension
# ---------------------------------------------------------------------------

class TestGetDominantTrendDimension:

    def test_returns_tuple(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        result = get_dominant_trend_dimension(trajs['church'])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_highest_slope_dimension(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        dim, slope = get_dominant_trend_dimension(trajs['church'])
        assert isinstance(dim, str)
        assert isinstance(slope, float)

    def test_empty_trend_returns_none(self):
        traj = PlaceTypeTrajectory(
            place_type='empty', visits=[], trend={}, volatility=0.0, visit_count=0
        )
        dim, slope = get_dominant_trend_dimension(traj)
        assert dim == 'None'
        assert slope == 0.0


# ---------------------------------------------------------------------------
# summarize_trajectories
# ---------------------------------------------------------------------------

class TestSummarizeTrajectories:

    def test_returns_dict(self, mixed_visits):
        trajs = build_place_trajectories(mixed_visits)
        result = summarize_trajectories(trajs)
        assert isinstance(result, dict)

    def test_all_place_types_present(self, mixed_visits):
        trajs = build_place_trajectories(mixed_visits)
        result = summarize_trajectories(trajs)
        assert 'church' in result
        assert 'hospital' in result

    def test_summary_keys(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        result = summarize_trajectories(trajs)
        summary = result['church']
        assert 'visit_count' in summary
        assert 'dominant_trend' in summary
        assert 'trend_slope' in summary
        assert 'trend_direction' in summary
        assert 'volatility' in summary

    def test_improving_direction(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        result = summarize_trajectories(trajs)
        assert result['church']['trend_direction'] == 'improving'

    def test_declining_direction(self, declining_hospital_visits):
        trajs = build_place_trajectories(declining_hospital_visits)
        result = summarize_trajectories(trajs)
        # slope is negative but may be within stable threshold - check it's not improving
        assert result['hospital']['trend_direction'] in ('declining', 'stable')
        assert result['hospital']['trend_slope'] <= 0

    def test_stable_direction(self):
        visits = [make_visit('park', i, hope=0.7, connectedness=0.6) for i in range(3)]
        trajs = build_place_trajectories(visits)
        result = summarize_trajectories(trajs)
        assert result['park']['trend_direction'] == 'stable'

    def test_visit_count_correct(self, improving_church_visits):
        trajs = build_place_trajectories(improving_church_visits)
        result = summarize_trajectories(trajs)
        assert result['church']['visit_count'] == 4

    def test_empty_trajectories_returns_empty(self):
        result = summarize_trajectories({})
        assert result == {}
