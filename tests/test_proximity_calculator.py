"""Tests for multi-dimensional proximity calculator.

Implements test cases PC-UT-001 through PC-UT-007 from TEST_PLAN.md.
"""

import pytest
import math
from dreams_app.core.extra.proximity_calculator import (
    categorical_proximity,
    linguistic_similarity,
    cultural_similarity,
    composite_proximity,
    normalize_geographic_distance,
)


class TestCategoricalProximity:
    """Test Case: PC-UT-002 - Categorical proximity - same type."""
    
    def test_same_type_perfect_match(self):
        """Same place types should return 1.0."""
        assert categorical_proximity('church', 'church') == 1.0
        assert categorical_proximity('hospital', 'hospital') == 1.0
        assert categorical_proximity('restaurant', 'restaurant') == 1.0
    
    def test_related_types_partial_match(self):
        """Related types (hospital/clinic) should return 0.5."""
        # PC-UT-003: Categorical proximity - related types
        assert categorical_proximity('hospital', 'clinic') == 0.5
        assert categorical_proximity('clinic', 'hospital') == 0.5
        assert categorical_proximity('church', 'temple') == 0.5
        assert categorical_proximity('restaurant', 'cafe') == 0.5
    
    def test_unrelated_types_no_match(self):
        """Unrelated types should return 0.0."""
        # PC-UT-004: Categorical proximity - unrelated types
        assert categorical_proximity('church', 'restaurant') == 0.0
        assert categorical_proximity('hospital', 'park') == 0.0
        assert categorical_proximity('school', 'museum') == 0.0
    
    def test_case_insensitive(self):
        """Type matching should be case-insensitive."""
        assert categorical_proximity('Church', 'church') == 1.0
        assert categorical_proximity('HOSPITAL', 'hospital') == 1.0
    
    def test_whitespace_handling(self):
        """Should handle leading/trailing whitespace."""
        assert categorical_proximity(' church ', 'church') == 1.0
        assert categorical_proximity('hospital', ' hospital ') == 1.0
    
    def test_empty_types(self):
        """Empty or None types should return 0.0."""
        assert categorical_proximity('', 'church') == 0.0
        assert categorical_proximity('church', '') == 0.0
        assert categorical_proximity('', '') == 0.0


class TestLinguisticSimilarity:
    """Test Case: PC-UT-005 - Linguistic similarity calculation."""
    
    def test_same_language_perfect_match(self):
        """Same language should return 1.0."""
        assert linguistic_similarity('portuguese', 'portuguese') == 1.0
        assert linguistic_similarity('english', 'english') == 1.0
        assert linguistic_similarity('spanish', 'spanish') == 1.0
    
    def test_different_languages_no_match(self):
        """Different languages should return 0.0."""
        assert linguistic_similarity('english', 'spanish') == 0.0
        assert linguistic_similarity('portuguese', 'french') == 0.0
    
    def test_case_insensitive(self):
        """Language matching should be case-insensitive."""
        assert linguistic_similarity('English', 'english') == 1.0
        assert linguistic_similarity('PORTUGUESE', 'portuguese') == 1.0
    
    def test_whitespace_handling(self):
        """Should handle leading/trailing whitespace."""
        assert linguistic_similarity(' english ', 'english') == 1.0
        assert linguistic_similarity('spanish', ' spanish ') == 1.0
    
    def test_empty_languages(self):
        """Empty or None languages should return 0.0."""
        assert linguistic_similarity('', 'english') == 0.0
        assert linguistic_similarity('english', '') == 0.0
        assert linguistic_similarity('', '') == 0.0


class TestCulturalSimilarity:
    """Test Case: PC-UT-006 - Cultural similarity (Jaccard index)."""
    
    def test_identical_tags_perfect_match(self):
        """Identical tag sets should return 1.0."""
        tags = ['european', 'catholic', 'traditional']
        assert cultural_similarity(tags, tags) == 1.0
        
        tags2 = ['a', 'b', 'c']
        assert cultural_similarity(tags2, tags2) == 1.0
    
    def test_partial_overlap(self):
        """Partial overlap should return correct Jaccard index."""
        # 2 common out of 4 total unique = 0.5
        tags1 = ['european', 'catholic', 'traditional']
        tags2 = ['european', 'traditional', 'historic']
        similarity = cultural_similarity(tags1, tags2)
        assert similarity == pytest.approx(0.5, abs=0.01)
        
        # 1 common out of 3 total unique = 0.333...
        tags3 = ['european', 'catholic']
        tags4 = ['european', 'traditional']
        similarity2 = cultural_similarity(tags3, tags4)
        assert similarity2 == pytest.approx(0.333, abs=0.01)
    
    def test_no_overlap(self):
        """No common tags should return 0.0."""
        tags1 = ['european', 'catholic']
        tags2 = ['asian', 'buddhist']
        assert cultural_similarity(tags1, tags2) == 0.0
    
    def test_empty_tags(self):
        """Empty tag lists should be handled correctly."""
        tags = ['european', 'catholic']
        assert cultural_similarity([], tags) == 0.0
        assert cultural_similarity(tags, []) == 0.0
        assert cultural_similarity([], []) == 1.0  # Both empty = identical
    
    def test_case_insensitive(self):
        """Tag matching should be case-insensitive."""
        tags1 = ['European', 'Catholic']
        tags2 = ['european', 'catholic']
        assert cultural_similarity(tags1, tags2) == 1.0
    
    def test_whitespace_handling(self):
        """Should handle whitespace in tags."""
        tags1 = [' european ', 'catholic']
        tags2 = ['european', ' catholic ']
        assert cultural_similarity(tags1, tags2) == 1.0


class TestCompositeProximity:
    """Test Case: PC-UT-007 - Composite proximity calculation."""
    
    def test_perfect_match_all_dimensions(self):
        """Perfect match across all dimensions should return 1.0."""
        place1 = {
            'type': 'church',
            'language': 'english',
            'cultural_tags': ['christian', 'traditional'],
            'geo_proximity': 1.0
        }
        place2 = {
            'type': 'church',
            'language': 'english',
            'cultural_tags': ['christian', 'traditional'],
            'geo_proximity': 1.0
        }
        
        result = composite_proximity(place1, place2)
        assert result == pytest.approx(1.0, abs=0.01)
    
    def test_weighted_sum_calculation(self):
        """Should calculate weighted sum correctly."""
        # Default weights: α=0.3, β=0.4, γ=0.15, δ=0.15
        place1 = {
            'type': 'church',
            'language': 'english',
            'cultural_tags': ['christian'],
            'geo_proximity': 1.0
        }
        place2 = {
            'type': 'hospital',  # cat=0.0
            'language': 'spanish',  # ling=0.0
            'cultural_tags': ['medical'],  # cult=0.0
            'geo_proximity': 1.0  # geo=1.0
        }
        
        # Expected: 0.3*1.0 + 0.4*0.0 + 0.15*0.0 + 0.15*0.0 = 0.3
        result = composite_proximity(place1, place2)
        assert result == pytest.approx(0.3, abs=0.01)
    
    def test_custom_weights(self):
        """Should accept custom weight configuration."""
        place1 = {
            'type': 'church',
            'language': 'english',
            'cultural_tags': ['christian'],
            'geo_proximity': 0.5
        }
        place2 = {
            'type': 'church',
            'language': 'english',
            'cultural_tags': ['christian'],
            'geo_proximity': 0.5
        }
        
        custom_weights = {'geo': 0.5, 'cat': 0.3, 'ling': 0.1, 'cult': 0.1}
        result = composite_proximity(place1, place2, weights=custom_weights)
        
        # Expected: 0.5*0.5 + 0.3*1.0 + 0.1*1.0 + 0.1*1.0 = 0.75
        assert result == pytest.approx(0.75, abs=0.01)
    
    def test_missing_attributes_default_to_zero(self):
        """Missing attributes should default gracefully."""
        place1 = {'type': 'church'}
        place2 = {'type': 'church'}
        
        # Should not crash, should use defaults
        result = composite_proximity(place1, place2)
        assert 0.0 <= result <= 1.0
    
    def test_partial_match_scenario(self):
        """Real-world scenario with partial matches."""
        place1 = {
            'type': 'hospital',
            'language': 'english',
            'cultural_tags': ['medical', 'modern'],
            'geo_proximity': 0.8
        }
        place2 = {
            'type': 'clinic',  # related, 0.5
            'language': 'english',  # same, 1.0
            'cultural_tags': ['medical', 'community'],  # 1/3 = 0.333
            'geo_proximity': 0.8
        }
        
        # Expected: 0.3*0.8 + 0.4*0.5 + 0.15*1.0 + 0.15*0.333 = 0.64
        result = composite_proximity(place1, place2)
        assert result == pytest.approx(0.64, abs=0.02)


class TestNormalizeGeographicDistance:
    """Test geographic distance normalization."""
    
    def test_zero_distance_perfect_proximity(self):
        """Zero distance should return 1.0."""
        assert normalize_geographic_distance(0.0) == 1.0
    
    def test_max_distance_near_zero_proximity(self):
        """Maximum distance should return near-zero proximity."""
        result = normalize_geographic_distance(500.0)
        assert 0.0 < result < 0.1
    
    def test_exponential_decay(self):
        """Should follow exponential decay pattern."""
        d1 = normalize_geographic_distance(50.0)
        d2 = normalize_geographic_distance(100.0)
        d3 = normalize_geographic_distance(200.0)
        
        # Each doubling should reduce proximity
        assert d1 > d2 > d3
    
    def test_negative_distance_raises_error(self):
        """Negative distance should raise ValueError."""
        with pytest.raises(ValueError, match="Distance cannot be negative"):
            normalize_geographic_distance(-10.0)
    
    def test_custom_max_distance(self):
        """Should accept custom maximum distance."""
        result = normalize_geographic_distance(100.0, max_distance_km=100.0)
        assert 0.0 < result < 0.2


class TestEdgeCases:
    """Edge case tests for proximity calculator."""
    
    def test_identical_locations_all_dimensions(self):
        """Identical locations should return maximum proximity."""
        # PC-EC-001 from TEST_PLAN
        place = {
            'type': 'church',
            'language': 'english',
            'cultural_tags': ['christian', 'traditional'],
            'geo_proximity': 1.0
        }
        
        result = composite_proximity(place, place)
        assert result == pytest.approx(1.0, abs=0.01)
    
    def test_completely_different_locations(self):
        """Completely different locations should return low proximity."""
        place1 = {
            'type': 'church',
            'language': 'english',
            'cultural_tags': ['christian'],
            'geo_proximity': 0.0
        }
        place2 = {
            'type': 'restaurant',
            'language': 'chinese',
            'cultural_tags': ['asian', 'food'],
            'geo_proximity': 0.0
        }
        
        result = composite_proximity(place1, place2)
        assert result == pytest.approx(0.0, abs=0.01)
    
    def test_unicode_in_tags(self):
        """Should handle Unicode characters in tags."""
        tags1 = ['café', 'français']
        tags2 = ['café', 'français']
        assert cultural_similarity(tags1, tags2) == 1.0
    
    def test_very_long_tag_lists(self):
        """Should handle large tag lists efficiently."""
        tags1 = [f'tag_{i}' for i in range(100)]
        tags2 = [f'tag_{i}' for i in range(50, 150)]
        
        # 50 common out of 150 total = 0.333...
        result = cultural_similarity(tags1, tags2)
        assert result == pytest.approx(0.333, abs=0.01)
