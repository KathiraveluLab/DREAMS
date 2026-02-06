# DREAMS Integration Guide - Location Proximity Module

## Overview

This guide provides step-by-step instructions for integrating the multi-dimensional location-proximity analysis module into the existing DREAMS platform. This work builds upon:

- **PR #77** (by kunal-595): EXIF GPS extraction already implemented in `dreamsApp/exif_extractor.py`
- **PR #70** (by AnvayKharb): Time-aware emotion proximity in `dreamsApp/analytics/emotion_proximity.py`

Our contribution adds **multi-dimensional spatial proximity** (geographic + categorical + linguistic + cultural) to complement the existing time-aware emotion analysis.

---

## Prerequisites

- DREAMS platform installed and running
- MongoDB instance configured
- Python 3.8+ environment
- Required packages: `exifread`, `scikit-learn`, `numpy`

---

## Integration Architecture

```
┌──────────────────────────────────────────────────────┐
│            Existing DREAMS Platform                   │
│                                                       │
│  ┌─────────────┐      ┌──────────────┐              │
│  │  Beehive    │─────▶│   Ingestion  │              │
│  │  Frontend   │      │   Pipeline   │              │
│  └─────────────┘      └──────┬───────┘              │
│                              │                       │
│                              ▼                       │
│  ┌──────────────────────────────────────┐           │
│  │      Sentiment Analysis              │           │
│  │      (existing)                      │           │
│  └──────────────┬───────────────────────┘           │
│                 │                                    │
└─────────────────┼────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│     NEW: Location-Proximity Module                   │
│                                                       │
│  ┌──────────────┐      ┌──────────────┐             │
│  │ EXIF         │─────▶│  Proximity   │             │
│  │ Extractor    │      │  Calculator  │             │
│  └──────────────┘      └──────┬───────┘             │
│                               │                      │
│                               ▼                      │
│  ┌──────────────────────────────────────┐           │
│  │   Emotion-Location Mapper            │           │
│  └──────────────┬───────────────────────┘           │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────┐               │
│  │   Semantic Clusterer             │               │
│  └──────────────┬───────────────────┘               │
└─────────────────┼────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│              MongoDB Storage                         │
│  - posts (extended with location)                   │
│  - location_analysis                                 │
│  - emotion_location_entries                          │
└──────────────────────────────────────────────────────┘
```

---

## Step 1: Extend Post Ingestion Route

**File**: `dreamsApp/app/ingestion/routes.py`

### 1.1 Import Location Modules

Add to the top of the file:
```python
from dreamsApp.exif_extractor import EXIFExtractor  # From PR #77 (kunal-595)
from dreamsApp.analytics.emotion_proximity import segment_timeline_into_windows  # From PR #70 (AnvayKharb)
from dreamsApp.location_proximity import extract_location, find_nearby_locations  # New multi-dimensional proximity
```

### 1.2 Modify Upload Route

Extend the existing `POST /upload` endpoint:

```python
@ingestion_bp.route('/upload', methods=['POST'])
def upload_photo():
    # Existing code for image upload and sentiment analysis...
    
    # Use existing EXIF extractor from PR #77
    extractor = EXIFExtractor()
    metadata = extractor.extract_metadata(image_path)
    location_data = metadata.get('location', {})
    
    # Fallback to manual location if no GPS in EXIF
    if location_data.get('accuracy') == 'none' and 'manual_location' in request.json:
        manual = request.json['manual_location']
        location_data = {
            'lat': manual['lat'],
            'lon': manual['lon'],
            'accuracy': 'manual'
        }
    
    # Store post with location
    post_doc = {
        'user_id': user_id,
        'caption': caption,
        'timestamp': datetime.utcnow(),
        'image_path': image_path,
        'sentiment': sentiment_result,
        'location': location_data  # NEW FIELD
    }
    
    post_id = db.posts.insert_one(post_doc).inserted_id
    
    # NEW: If location available, find nearby locations and update analysis
    if location_data.get('lat') and location_data.get('lon'):
        from location_proximity.emotion_location_mapper import EmotionLocationMapper
        
        mapper = EmotionLocationMapper()
        
        # Add emotion-location entry
        mapper.add_entry(
            location_id=str(post_id),  # Use post_id as location_id initially
            sentiment=sentiment_result['label'],
            score=sentiment_result['score'],
            metadata={
                'timestamp': post_doc['timestamp'],
                'coordinates': location_data,
                'user_id': user_id
            }
        )
        
        # Store in emotion_location_entries collection
        db.emotion_location_entries.insert_one({
            'user_id': user_id,
            'location_id': str(post_id),
            'post_id': post_id,
            'sentiment': sentiment_result['label'],
            'score': sentiment_result['score'],
            'timestamp': post_doc['timestamp'],
            'coordinates': location_data
        })
        
        # Find nearby locations
        user_locations = list(db.posts.find({
            'user_id': user_id,
            'location.lat': {'$exists': True}
        }))
        
        nearby = find_nearby_locations(
            target_location={'lat': location_data['lat'], 'lon': location_data['lon']},
            locations=[
                {'lat': loc['location']['lat'], 'lon': loc['location']['lon']}
                for loc in user_locations
            ],
            radius_meters=1000  # 1km radius
        )
        
        # Update location_analysis collection
        db.location_analysis.update_one(
            {'user_id': user_id},
            {
                '$push': {
                    'locations': {
                        'id': str(post_id),
                        'coordinates': location_data,
                        'timestamp': post_doc['timestamp'],
                        'sentiment': sentiment_result['label'],
                        'nearby_count': len(nearby)
                    }
                },
                '$set': {'updated_at': datetime.utcnow()}
            },
            upsert=True
        )
    
    return jsonify({
        'post_id': str(post_id),
        'sentiment': sentiment_result,
        'location': location_data,
        'nearby_locations': len(nearby) if location_data.get('lat') else 0
    })
```

---

## Step 2: Create Location Analysis Dashboard Route

**File**: `dreamsApp/app/dashboard/main.py`

### 2.1 Add New Route

```python
from flask import render_template
from location_proximity.semantic_clustering import SemanticLocationClusterer
from location_proximity.emotion_location_mapper import EmotionLocationMapper

@dashboard_bp.route('/location_analysis/<user_id>')
def location_analysis(user_id):
    """Display location-emotion analysis dashboard."""
    
    # Get user's location data
    analysis_doc = db.location_analysis.find_one({'user_id': user_id})
    
    if not analysis_doc:
        return render_template('dashboard/location_analysis.html', 
                             error="No location data available")
    
    # Get emotion-location entries
    entries = list(db.emotion_location_entries.find({'user_id': user_id}))
    
    # Initialize mapper and load data
    mapper = EmotionLocationMapper()
    for entry in entries:
        mapper.add_entry(
            location_id=entry['location_id'],
            sentiment=entry['sentiment'],
            score=entry['score'],
            metadata={
                'timestamp': entry['timestamp'],
                'coordinates': entry['coordinates']
            }
        )
    
    # Find hotspots
    positive_hotspots = mapper.find_emotional_hotspots('positive', min_visits=3)
    negative_hotspots = mapper.find_emotional_hotspots('negative', min_visits=3)
    
    # Perform clustering if enough locations
    clusters = []
    if len(analysis_doc.get('locations', [])) >= 6:
        # Build proximity matrix (simplified - use actual multi-dimensional in production)
        from location_proximity.proximity_calculator import compute_proximity_matrix
        
        locations = analysis_doc['locations']
        proximity_matrix = compute_proximity_matrix(locations)
        
        clusterer = SemanticLocationClusterer(eps=0.4, min_samples=2)
        cluster_labels = clusterer.cluster_by_proximity(proximity_matrix)
        
        # Get emotion profiles for clusters
        clusters = clusterer.cluster_with_emotions(proximity_matrix, entries)
    
    return render_template('dashboard/location_analysis.html',
                          user_id=user_id,
                          locations=analysis_doc.get('locations', []),
                          positive_hotspots=positive_hotspots,
                          negative_hotspots=negative_hotspots,
                          clusters=clusters,
                          total_locations=len(analysis_doc.get('locations', [])))
```

### 2.2 Create Template

**File**: `dreamsApp/app/templates/dashboard/location_analysis.html`

```html
{% extends "base.html" %}

{% block content %}
<div class="container">
    <h1>Location-Emotion Analysis</h1>
    
    {% if error %}
        <div class="alert alert-warning">{{ error }}</div>
    {% else %}
        <!-- Summary Stats -->
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5>Total Locations</h5>
                        <p class="display-4">{{ total_locations }}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5>Positive Hotspots</h5>
                        <p class="display-4">{{ positive_hotspots|length }}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5>Negative Hotspots</h5>
                        <p class="display-4">{{ negative_hotspots|length }}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5>Clusters</h5>
                        <p class="display-4">{{ clusters|length }}</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Map placeholder -->
        <div class="row mt-4">
            <div class="col-12">
                <h3>Emotional Hotspots Map</h3>
                <div id="map" style="height: 400px; background: #f0f0f0;">
                    <!-- Leaflet.js map integration here -->
                </div>
            </div>
        </div>
        
        <!-- Clusters -->
        <div class="row mt-4">
            <div class="col-12">
                <h3>Location Clusters</h3>
                {% for cluster in clusters %}
                <div class="card mb-3">
                    <div class="card-header">
                        Cluster {{ cluster.cluster_id }}: {{ cluster.label }}
                    </div>
                    <div class="card-body">
                        <p><strong>Members:</strong> {{ cluster.members|length }}</p>
                        <p><strong>Emotion Profile:</strong></p>
                        <div class="progress">
                            <div class="progress-bar bg-success" style="width: {{ cluster.emotion_distribution.positive * 100 }}%">
                                Positive {{ "%.0f"|format(cluster.emotion_distribution.positive * 100) }}%
                            </div>
                            <div class="progress-bar bg-warning" style="width: {{ cluster.emotion_distribution.neutral * 100 }}%">
                                Neutral {{ "%.0f"|format(cluster.emotion_distribution.neutral * 100) }}%
                            </div>
                            <div class="progress-bar bg-danger" style="width: {{ cluster.emotion_distribution.negative * 100 }}%">
                                Negative {{ "%.0f"|format(cluster.emotion_distribution.negative * 100) }}%
                            </div>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    {% endif %}
</div>
{% endblock %}
```

---

## Step 3: Database Indexes

Add indexes for efficient querying:

```python
# In a migration script or app initialization
db.posts.create_index([('user_id', 1), ('location.lat', 1)])
db.emotion_location_entries.create_index([('user_id', 1), ('timestamp', -1)])
db.emotion_location_entries.create_index([('user_id', 1), ('location_id', 1)])
db.location_analysis.create_index([('user_id', 1)])
```

---

## Step 4: Configuration

**File**: `dreamsApp/app/config.py`

Add location-proximity settings:

```python
class Config:
    # Existing config...
    
    # Location-Proximity Settings
    LOCATION_PROXIMITY_WEIGHTS = {
        'geographic': 0.3,
        'categorical': 0.4,
        'linguistic': 0.15,
        'cultural': 0.15
    }
    
    CLUSTERING_PARAMS = {
        'eps': 0.4,
        'min_samples': 2
    }
    
    HOTSPOT_MIN_VISITS = 3
    HOTSPOT_MIN_CONFIDENCE = 0.6
    
    NEARBY_RADIUS_METERS = 1000
```

---

## Step 5: Testing Integration

Create integration test:

**File**: `tests/test_location_emotion_integration.py`

```python
import pytest
import json
from dreamsApp.app import create_app

def test_full_pipeline():
    """Test full pipeline: upload → location extraction → emotion mapping → clustering."""
    
    app = create_app('testing')
    client = app.test_client()
    
    # Load test data
    with open('tests/data/locations.json') as f:
        test_locations = json.load(f)['locations']
    
    # Simulate uploads for multiple locations
    for loc in test_locations[:5]:
        response = client.post('/upload', json={
            'user_id': 'test_user',
            'image': 'base64_image_here',
            'caption': f'Visit to {loc["name"]}',
            'manual_location': loc['coordinates']
        })
        
        assert response.status_code == 200
        data = response.json
        assert 'location' in data
        assert data['location']['lat'] == loc['coordinates']['lat']
    
    # Check location analysis was created
    response = client.get('/dashboard/test_user/location_analysis')
    assert response.status_code == 200
```

---

## Step 6: Deployment Checklist

- [ ] Install required packages: `pip install exifread scikit-learn`
- [ ] Create MongoDB indexes
- [ ] Update `requirements.txt`
- [ ] Add location-proximity settings to config
- [ ] Extend ingestion route with location extraction
- [ ] Create location analysis dashboard route and template
- [ ] Run integration tests
- [ ] Update API documentation
- [ ] Deploy to staging environment
- [ ] Monitor performance and errors

---

## Troubleshooting

### Issue: No GPS data in uploaded images

**Solution**: Ensure fallback to manual location:
```python
if 'manual_location' in request.json:
    location_data = request.json['manual_location']
```

### Issue: Clustering fails with too few locations

**Solution**: Add minimum check:
```python
if len(locations) < 6:
    return {'error': 'Need at least 6 locations for clustering'}
```

### Issue: Slow proximity calculations

**Solution**: Implement caching:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_proximity(loc1_id, loc2_id):
    return compute_proximity(loc1, loc2)
```

---

## Next Steps

1. **Add Place Type Inference**: Use Google Places API to automatically tag locations
2. **Implement Real-time Updates**: WebSocket support for live clustering
3. **Cross-User Analysis**: Privacy-preserving aggregation of emotion patterns
4. **Mobile Support**: Optimize for mobile dashboard viewing

---

**Integration Version**: 1.0  
**Last Updated**: February 3, 2026  
**Author**: Krishan (GSoC 2026 Contributor)
