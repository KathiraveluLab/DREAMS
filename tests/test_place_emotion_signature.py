import math
import pytest
from dreams_app.core.extra.place_emotion_signature import (
    CHIME_DIMENSIONS,
    PlaceEmotionSignature,
    build_place_signature,
    chime_proximity,
    detect_recovery_anchors,
    get_dominant_chime_dimension,
)


@pytest.fixture
def church_visits():
    return [
        {'Connectedness': 0.7, 'Hope': 0.9, 'Identity': 0.3, 'Meaning': 0.6, 'Empowerment': 0.4},
        {'Connectedness': 0.75, 'Hope': 0.85, 'Identity': 0.35, 'Meaning': 0.65, 'Empowerment': 0.45},
        {'Connectedness': 0.72, 'Hope': 0.88, 'Identity': 0.32, 'Meaning': 0.62, 'Empowerment': 0.42},
    ]


@pytest.fixture
def community_center_visits():
    return [
        {'Connectedness': 0.82, 'Hope': 0.79, 'Identity': 0.4, 'Meaning': 0.55, 'Empowerment': 0.5},
        {'Connectedness': 0.85, 'Hope': 0.81, 'Identity': 0.38, 'Meaning': 0.58, 'Empowerment': 0.52},
    ]


@pytest.fixture
def hospital_visits():
    return [
        {'Connectedness': 0.2, 'Hope': 0.3, 'Identity': 0.6, 'Meaning': 0.4, 'Empowerment': 0.25},
        {'Connectedness': 0.25, 'Hope': 0.28, 'Identity': 0.55, 'Meaning': 0.45, 'Empowerment': 0.3},
    ]


@pytest.fixture
def church_sig(church_visits):
    return build_place_signature('church', church_visits)


@pytest.fixture
def community_sig(community_center_visits):
    return build_place_signature('community_center', community_center_visits)


@pytest.fixture
def hospital_sig(hospital_visits):
    return build_place_signature('hospital', hospital_visits)


class TestBuildPlaceSignature:

    def test_returns_place_emotion_signature(self, church_visits):
        sig = build_place_signature('church', church_visits)
        assert isinstance(sig, PlaceEmotionSignature)

    def test_place_type_stored(self, church_visits):
        sig = build_place_signature('church', church_visits)
        assert sig.place_type == 'church'

    def test_visit_count(self, church_visits):
        sig = build_place_signature('church', church_visits)
        assert sig.visit_count == 3

    def test_mean_vector_computed(self, church_visits):
        sig = build_place_signature('church', church_visits)
        expected_hope = (0.9 + 0.85 + 0.88) / 3
        assert abs(sig.chime_vector['Hope'] - expected_hope) < 1e-9

    def test_all_chime_dimensions_present(self, church_visits):
        sig = build_place_signature('church', church_visits)
        for dim in CHIME_DIMENSIONS:
            assert dim in sig.chime_vector

    def test_volatility_is_non_negative(self, church_visits):
        sig = build_place_signature('church', church_visits)
        assert sig.volatility >= 0.0

    def test_low_volatility_for_consistent_visits(self):
        visits = [
            {'Connectedness': 0.7, 'Hope': 0.9, 'Identity': 0.3, 'Meaning': 0.6, 'Empowerment': 0.4}
        ] * 5
        sig = build_place_signature('church', visits)
        assert sig.volatility == 0.0

    def test_high_volatility_for_inconsistent_visits(self):
        visits = [
            {'Connectedness': 0.1, 'Hope': 0.1, 'Identity': 0.1, 'Meaning': 0.1, 'Empowerment': 0.1},
            {'Connectedness': 0.9, 'Hope': 0.9, 'Identity': 0.9, 'Meaning': 0.9, 'Empowerment': 0.9},
        ]
        sig = build_place_signature('volatile_place', visits)
        assert sig.volatility > 0.3

    def test_empty_visits_returns_zero_vector(self):
        sig = build_place_signature('unknown', [])
        assert sig.visit_count == 0
        for dim in CHIME_DIMENSIONS:
            assert sig.chime_vector[dim] == 0.0

    def test_empty_visits_zero_volatility(self):
        sig = build_place_signature('unknown', [])
        assert sig.volatility == 0.0

    def test_single_visit(self):
        visits = [{'Connectedness': 0.8, 'Hope': 0.7, 'Identity': 0.5, 'Meaning': 0.6, 'Empowerment': 0.4}]
        sig = build_place_signature('park', visits)
        assert sig.visit_count == 1
        assert sig.chime_vector['Hope'] == 0.7
        assert sig.volatility == 0.0

    def test_missing_dimension_defaults_to_zero(self):
        visits = [{'Hope': 0.8}]
        sig = build_place_signature('partial', visits)
        assert sig.chime_vector['Hope'] == 0.8
        assert sig.chime_vector['Connectedness'] == 0.0


class TestChimeProximity:

    def test_identical_signatures_return_one(self, church_visits):
        sig = build_place_signature('church', church_visits)
        assert abs(chime_proximity(sig, sig) - 1.0) < 1e-9

    def test_similar_signatures_high_proximity(self, church_sig, community_sig):
        score = chime_proximity(church_sig, community_sig)
        assert score > 0.8

    def test_dissimilar_signatures_lower_proximity(self, church_sig, community_sig, hospital_sig):
        assert chime_proximity(church_sig, hospital_sig) < chime_proximity(church_sig, community_sig)

    def test_proximity_range(self, church_sig, hospital_sig):
        score = chime_proximity(church_sig, hospital_sig)
        assert 0.0 <= score <= 1.0

    def test_symmetry(self, church_sig, hospital_sig):
        assert abs(
            chime_proximity(church_sig, hospital_sig) -
            chime_proximity(hospital_sig, church_sig)
        ) < 1e-9

    def test_zero_vector_returns_zero(self):
        zero_sig = PlaceEmotionSignature('empty', {d: 0.0 for d in CHIME_DIMENSIONS}, visit_count=1)
        other_sig = PlaceEmotionSignature('other', {'Hope': 0.9, 'Connectedness': 0.8, 'Identity': 0.3, 'Meaning': 0.5, 'Empowerment': 0.4}, visit_count=1)
        assert chime_proximity(zero_sig, other_sig) == 0.0

    def test_insufficient_visits_returns_neutral(self):
        sig_a = PlaceEmotionSignature('a', {'Hope': 0.9, 'Connectedness': 0.8, 'Identity': 0.3, 'Meaning': 0.5, 'Empowerment': 0.4}, visit_count=0)
        sig_b = PlaceEmotionSignature('b', {'Hope': 0.1, 'Connectedness': 0.2, 'Identity': 0.8, 'Meaning': 0.3, 'Empowerment': 0.2}, visit_count=3)
        assert chime_proximity(sig_a, sig_b, min_visits=1) == 0.5

    def test_min_visits_threshold(self, church_sig):
        low_visit_sig = PlaceEmotionSignature('rare', {'Hope': 0.9, 'Connectedness': 0.8, 'Identity': 0.3, 'Meaning': 0.5, 'Empowerment': 0.4}, visit_count=1)
        assert chime_proximity(church_sig, low_visit_sig, min_visits=2) == 0.5
        assert chime_proximity(church_sig, low_visit_sig, min_visits=1) != 0.5


class TestDetectRecoveryAnchors:

    @pytest.fixture
    def signatures(self, church_sig, community_sig, hospital_sig):
        return {'church': church_sig, 'community_center': community_sig, 'hospital': hospital_sig}

    def test_returns_list(self, signatures):
        assert isinstance(detect_recovery_anchors(signatures), list)

    def test_anchor_tuple_structure(self, signatures):
        for item in detect_recovery_anchors(signatures, threshold=0.0):
            place_type, score, dimension = item
            assert isinstance(place_type, str)
            assert isinstance(score, float)
            assert isinstance(dimension, str)

    def test_sorted_by_score_descending(self, signatures):
        result = detect_recovery_anchors(signatures, threshold=0.0)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_church_is_top_anchor(self, signatures):
        result = detect_recovery_anchors(signatures, threshold=0.0)
        if result:
            assert result[0][0] in ('church', 'community_center')

    def test_hospital_not_anchor_with_high_threshold(self, signatures):
        result = detect_recovery_anchors(signatures, threshold=0.7)
        assert 'hospital' not in [r[0] for r in result]

    def test_threshold_filters_correctly(self, signatures):
        all_results = detect_recovery_anchors(signatures, threshold=0.0)
        filtered = detect_recovery_anchors(signatures, threshold=0.9)
        assert len(filtered) <= len(all_results)

    def test_min_visits_filters_single_visit(self):
        sigs = {
            'church': PlaceEmotionSignature('church', {'Hope': 0.9, 'Connectedness': 0.8, 'Identity': 0.3, 'Meaning': 0.5, 'Empowerment': 0.4}, visit_count=1),
            'park': PlaceEmotionSignature('park', {'Hope': 0.85, 'Connectedness': 0.75, 'Identity': 0.35, 'Meaning': 0.55, 'Empowerment': 0.45}, visit_count=3),
        }
        result = detect_recovery_anchors(sigs, threshold=0.0, min_visits=2)
        assert 'church' not in [r[0] for r in result]
        assert 'park' in [r[0] for r in result]

    def test_empty_signatures_returns_empty(self):
        assert detect_recovery_anchors({}) == []

    def test_custom_anchor_dimensions(self, signatures):
        result = detect_recovery_anchors(signatures, anchor_dimensions=['Meaning', 'Identity'], threshold=0.0)
        for _, _, dim in result:
            assert dim in ['Meaning', 'Identity']

    def test_volatility_penalizes_anchor_score(self):
        stable = PlaceEmotionSignature('stable', {'Hope': 0.9, 'Connectedness': 0.8, 'Identity': 0.3, 'Meaning': 0.5, 'Empowerment': 0.4}, visit_count=3, volatility=0.0)
        volatile = PlaceEmotionSignature('volatile', {'Hope': 0.9, 'Connectedness': 0.8, 'Identity': 0.3, 'Meaning': 0.5, 'Empowerment': 0.4}, visit_count=3, volatility=0.8)
        result = detect_recovery_anchors({'stable': stable, 'volatile': volatile}, threshold=0.0)
        scores = {r[0]: r[1] for r in result}
        assert scores.get('stable', 0) > scores.get('volatile', 0)


class TestGetDominantChimeDimension:

    def test_returns_tuple(self, church_sig):
        assert isinstance(get_dominant_chime_dimension(church_sig), tuple)

    def test_returns_highest_dimension(self):
        sig = PlaceEmotionSignature('test', {'Connectedness': 0.3, 'Hope': 0.9, 'Identity': 0.2, 'Meaning': 0.5, 'Empowerment': 0.4}, visit_count=2)
        dim, score = get_dominant_chime_dimension(sig)
        assert dim == 'Hope'
        assert score == 0.9

    def test_empty_vector_returns_none(self):
        sig = PlaceEmotionSignature('empty', {}, visit_count=0)
        dim, score = get_dominant_chime_dimension(sig)
        assert dim == 'None'
        assert score == 0.0

    def test_church_dominant_is_hope(self, church_sig):
        dim, score = get_dominant_chime_dimension(church_sig)
        assert dim == 'Hope'
        assert score > 0.8
