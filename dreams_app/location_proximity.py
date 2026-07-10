"""Location proximity analysis module for geographic calculations and clustering."""

import math
from collections import deque
from typing import List, Dict, Optional, TypedDict


class Location(TypedDict):
    """Location data structure."""
    lat: float
    lon: float


class ProximityResult(TypedDict):
    """Proximity calculation result."""
    distance: float
    is_proximate: bool


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate GPS coordinates are within valid ranges.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates using Haversine formula.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        
    Returns:
        Distance in meters between the two points
    """
    if not (validate_coordinates(lat1, lon1) and validate_coordinates(lat2, lon2)):
        raise ValueError("Invalid coordinates")
        
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def compute_proximity(location1: Location, location2: Location, threshold_meters: float) -> ProximityResult:
    """Compute proximity between two geographic locations.
    
    Args:
        location1: First location with lat/lon coordinates
        location2: Second location with lat/lon coordinates  
        threshold_meters: Distance threshold in meters for proximity detection
        
    Returns:
        Dictionary with distance and proximity boolean result
    """
    dist = calculate_distance(location1["lat"], location1["lon"], location2["lat"], location2["lon"])
    return {
        "distance": dist,
        "is_proximate": dist <= threshold_meters
    }


def extract_location(metadata: Dict) -> Optional[Location]:
    """Extract location data from photo metadata.
    
    Args:
        metadata: Photo metadata dictionary containing location information
        
    Returns:
        Dictionary with lat/lon coordinates and accuracy, or None if no location data
    """
    loc = metadata.get("location")
    if not isinstance(loc, dict):
        return None
        
    if "lat" in loc and "lon" in loc:
        try:
            lat = float(loc["lat"])
            lon = float(loc["lon"])
            if validate_coordinates(lat, lon):
                return {"lat": lat, "lon": lon}
        except (ValueError, TypeError):
            pass
    return None


def find_nearby_locations(target_location: Location, locations: List[Location], 
                         radius_meters: float) -> List[Location]:
    """Find all locations within specified radius of target location.
    
    Args:
        target_location: Reference location with lat/lon coordinates
        locations: List of locations to search through
        radius_meters: Search radius in meters
        
    Returns:
        List of locations within the specified radius
    """
    nearby = []
    for loc in locations:
        if loc == target_location:
            continue
        try:
            prox = compute_proximity(target_location, loc, radius_meters)
            if prox["is_proximate"]:
                nearby.append(loc)
        except ValueError:
            pass
    return nearby


def cluster_locations(locations: List[Location], proximity_threshold: float) -> List[List[Location]]:
    """Cluster locations based on geographic proximity.
    
    Args:
        locations: List of location dictionaries with coordinates
        proximity_threshold: Distance threshold in meters for clustering
        
    Returns:
        List of location clusters, each cluster is a list of nearby locations
    """
    clusters = []
    visited = set()
    
    for i in range(len(locations)):
        if i in visited:
            continue
        
        cluster = [locations[i]]
        visited.add(i)
        
        # Breadth-first search to find all connected locations
        queue = deque([i])
        while queue:
            curr_idx = queue.popleft()
            for j in range(len(locations)):
                if j not in visited:
                    try:
                        prox = compute_proximity(locations[curr_idx], locations[j], proximity_threshold)
                        if prox["is_proximate"]:
                            visited.add(j)
                            cluster.append(locations[j])
                            queue.append(j)
                    except ValueError:
                        pass
        clusters.append(cluster)
        
    return clusters