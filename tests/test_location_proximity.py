"""Tests for location proximity analysis module."""

import pytest
from dreamsApp.location_proximity import (
    extract_location,
    compute_proximity,
    cluster_locations,
    calculate_distance,
    validate_coordinates,
    find_nearby_locations,
    Location
)


class TestLocationProximity:
    """Test cases for geographic location proximity functions."""
    
    def test_extract_location(self):
        """Test safe extraction from metadata dictionaries."""
        metadata = {"location": {"lat": 61.2181, "lon": -149.9003}}
        result = extract_location(metadata)
        assert result == {"lat": 61.2181, "lon": -149.9003}
        
        assert extract_location({}) is None
        assert extract_location({"location": "not_dict"}) is None
        assert extract_location({"location": {"lat": "invalid", "lon": 10}}) is None
        assert extract_location({"location": {"lat": 91.0, "lon": 10}}) is None
    
    def test_compute_proximity(self):
        """Test boolean proximity boundaries."""
        loc1: Location = {"lat": 61.2181, "lon": -149.9003}
        loc2: Location = {"lat": 61.2182, "lon": -149.9004}
        
        res = compute_proximity(loc1, loc2, 1000.0)
        assert res["is_proximate"] is True
        assert res["distance"] > 0
        
        res_strict = compute_proximity(loc1, loc2, 1.0)
        assert res_strict["is_proximate"] is False
    
    def test_cluster_locations(self):
        """Test BFS Connected Component grouping on distance threshold."""
        locations: list[Location] = [
            {"lat": 61.2181, "lon": -149.9003},
            {"lat": 61.2182, "lon": -149.9004},
            {"lat": 34.0522, "lon": -118.2437}  # Much farther
        ]
        
        clusters = cluster_locations(locations, 1000.0)
        assert len(clusters) == 2
        sizes = sorted([len(c) for c in clusters])
        assert sizes == [1, 2]
    
    def test_calculate_distance(self):
        """Test Haversine distance returns approximate correct meter distance."""
        dist = calculate_distance(61.2181, -149.9003, 61.2182, -149.9004)
        assert 10 < dist < 20 # Approx 12.3m
        
        with pytest.raises(ValueError):
            calculate_distance(91.0, 0.0, 0.0, 0.0)
    
    def test_validate_coordinates(self):
        """Test strict coordinate ranges."""
        assert validate_coordinates(61.2181, -149.9003) is True
        assert validate_coordinates(91.0, 0.0) is False
        assert validate_coordinates(-90.1, 0.0) is False
        assert validate_coordinates(0.0, 181.0) is False
        assert validate_coordinates(0.0, -181.0) is False
    
    def test_find_nearby_locations(self):
        """Test radial querying excluding out of bounds and self."""
        target: Location = {"lat": 61.2181, "lon": -149.9003}
        locations: list[Location] = [
            {"lat": 61.2182, "lon": -149.9004},
            {"lat": 34.0522, "lon": -118.2437},
            target
        ]
        
        nearby = find_nearby_locations(target, locations, 1000.0)
        assert len(nearby) == 1
        assert nearby[0] == locations[0]