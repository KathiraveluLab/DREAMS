# DREAMS API Design - Location Proximity & Emotion Analysis

## Overview

This document outlines the REST API design for multi-dimensional location-proximity analysis within DREAMS. The API builds upon:

- **Existing EXIF extraction** (PR #77 by kunal-595): GPS coordinate extraction from image metadata
- **Existing emotion proximity** (PR #70 by AnvayKharb): Time-aware emotion timeline comparison

Our API adds **spatial proximity endpoints** for geographic clustering, place-type similarity, and emotion-location mapping.

---

## API Endpoints

### 1. Ingestion & Analysis Endpoints

#### POST `/api/upload`
**Description**: Upload photo with caption, extract location, analyze sentiment, and compute proximity patterns.

**Request**:
```json
{
  "user_id": "string",
  "image": "base64_encoded_image",
  "caption": "string",
  "timestamp": "ISO8601_datetime",
  "manual_location": {  // Optional fallback if no EXIF GPS
    "lat": 61.2181,
    "lon": -149.9003
  }
}
```

**Response**:
```json
{
  "post_id": "string",
  "sentiment": {
    "label": "positive|neutral|negative",
    "score": 0.85
  },
  "location": {
    "lat": 61.2181,
    "lon": -149.9003,
    "accuracy": "high|medium|low|none",
    "place_type": "park",  // Inferred or manual
    "nearby_locations": [
      {
        "location_id": "string",
        "distance_meters": 150.5,
        "proximity_score": 0.75
      }
    ]
  },
  "keywords": ["keyword1", "keyword2"],
  "processing_time_ms": 1234
}
```

**Integration Point**: `dreamsApp/app/ingestion/routes.py`

---

#### GET `/api/location/proximity`
**Description**: Calculate multi-dimensional proximity between two locations.

**Query Parameters**:
- `location1_id` (string): First location ID
- `location2_id` (string): Second location ID
- `weights` (optional string): JSON object `{"geo": 0.3, "cat": 0.4, "ling": 0.15, "cult": 0.15}`

**Response**:
```json
{
  "location1": {
    "id": "park_001",
    "name": "Delaney Park Strip",
    "type": "park"
  },
  "location2": {
    "id": "park_002",
    "name": "Chugach State Park",
    "type": "park"
  },
  "proximity_scores": {
    "geographic": 0.45,
    "categorical": 1.0,
    "linguistic": 1.0,
    "cultural": 0.67,
    "composite": 0.78
  },
  "distance_meters": 8542.3
}
```

---

#### POST `/api/location/cluster`
**Description**: Cluster user's locations using multi-dimensional proximity.

**Request**:
```json
{
  "user_id": "string",
  "method": "dbscan|kmeans",
  "params": {
    "eps": 0.4,
    "min_samples": 2
  }
}
```

**Response**:
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "label": "Parks & Recreation",
      "members": ["park_001", "park_002", "park_003"],
      "centroid": {"lat": 61.19, "lon": -149.88},
      "emotion_profile": {
        "positive": 0.75,
        "neutral": 0.15,
        "negative": 0.10
      }
    }
  ],
  "noise_points": ["location_xyz"],
  "silhouette_score": 0.68
}
```

---

### 2. Emotion-Location Query Endpoints

#### GET `/api/location/{location_id}/emotions`
**Description**: Get emotion profile for a specific location.

**Response**:
```json
{
  "location_id": "church_001",
  "name": "St. Mary's Catholic Church",
  "total_visits": 5,
  "emotion_distribution": {
    "positive": 0.80,
    "neutral": 0.15,
    "negative": 0.05
  },
  "mean_score": 0.82,
  "timeline": [
    {
      "timestamp": "2024-01-21T10:00:00Z",
      "sentiment": "positive",
      "score": 0.88
    }
  ]
}
```

---

#### GET `/api/location/hotspots`
**Description**: Find emotional hotspots for a user.

**Query Parameters**:
- `user_id` (string): User ID
- `sentiment` (string): Filter by `positive|neutral|negative`
- `min_confidence` (float): Minimum confidence threshold (default 0.6)
- `min_visits` (int): Minimum visits required (default 3)

**Response**:
```json
{
  "hotspots": [
    {
      "location_id": "church_001",
      "name": "St. Mary's Catholic Church",
      "sentiment": "positive",
      "confidence": 0.80,
      "visit_count": 5,
      "coordinates": {"lat": 61.2167, "lon": -149.8944}
    }
  ]
}
```

---

#### GET `/api/location/place-type-comparison`
**Description**: Compare emotions across place types.

**Query Parameters**:
- `user_id` (string): User ID

**Response**:
```json
{
  "place_types": {
    "church": {
      "positive": 0.85,
      "neutral": 0.10,
      "negative": 0.05,
      "mean_score": 0.82,
      "visit_count": 8
    },
    "hospital": {
      "positive": 0.15,
      "neutral": 0.20,
      "negative": 0.65,
      "mean_score": 0.31,
      "visit_count": 6
    },
    "park": {
      "positive": 0.70,
      "neutral": 0.20,
      "negative": 0.10,
      "mean_score": 0.75,
      "visit_count": 10
    }
  }
}
```

---

### 3. Dashboard Visualization Endpoints

#### GET `/api/dashboard/{user_id}/location-analysis`
**Description**: Get comprehensive location analysis for dashboard.

**Response**:
```json
{
  "summary": {
    "total_locations": 24,
    "unique_place_types": 5,
    "clusters": 4,
    "hotspots": 3
  },
  "clusters": [...],  // Same as cluster endpoint
  "hotspots": [...],  // Same as hotspots endpoint
  "temporal_patterns": {
    "weekly_distribution": {
      "Monday": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
      "Tuesday": {...}
    },
    "place_type_evolution": [
      {
        "week": "2024-W01",
        "church": {"positive": 0.8},
        "hospital": {"negative": 0.6}
      }
    ]
  }
}
```

---

## Data Flow Architecture

```
┌─────────────────┐
│  Photo Upload   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EXIF Extraction │ ──► GPS Coordinates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sentiment       │ ──► Emotion Score
│ Analysis        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Location-Emotion Mapper         │
│ - Store location + emotion pair │
│ - Update visit history          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Proximity Calculation           │
│ - Find nearby locations         │
│ - Compute multi-dim scores      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Pattern Detection               │
│ - Identify hotspots             │
│ - Cluster analysis              │
│ - Temporal trends               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ MongoDB Storage │
└─────────────────┘
```

---

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional context"
    }
  }
}
```

### Error Codes
- `INVALID_IMAGE_FORMAT`: Unsupported image format
- `NO_GPS_DATA`: No GPS coordinates in EXIF or manual location
- `LOCATION_NOT_FOUND`: Location ID doesn't exist
- `INSUFFICIENT_DATA`: Not enough data for clustering/analysis
- `INVALID_COORDINATES`: GPS coordinates out of valid range
- `PROCESSING_FAILED`: General processing error

---

## Rate Limiting

- **Upload**: 10 requests/minute per user
- **Query endpoints**: 100 requests/minute per user
- **Dashboard**: 20 requests/minute per user

Headers:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1640995200
```

---

## Authentication

All endpoints require Bearer token authentication:
```
Authorization: Bearer <token>
```

User context is extracted from the JWT token. The `user_id` in requests must match the authenticated user (except for admin users).

---

## Integration Points

### Existing DREAMS Components

1. **Ingestion Pipeline** (`dreamsApp/app/ingestion/routes.py`)
   - Extend `POST /upload` to include location extraction
   - Add location-proximity calculations after sentiment analysis

2. **Dashboard** (`dreamsApp/app/dashboard/main.py`)
   - Add new route `/location_analysis/<user_id>`
   - Integrate location map visualization
   - Display cluster cards and hotspot markers

3. **Data Models** (`dreamsApp/app/models.py`)
   - Extend `Post` model with location fields
   - Add `LocationAnalysis` model for storing clusters/hotspots
   - Add `EmotionLocationEntry` for tracking location-emotion pairs

### New Components

1. **Location Proximity Calculator** (`location_proximity/proximity_calculator.py`)
   - Called by API endpoints to compute multi-dimensional scores

2. **Emotion-Location Mapper** (`location_proximity/emotion_location_mapper.py`)
   - Manages emotion-location associations
   - Provides hotspot detection and pattern analysis

3. **Semantic Clusterer** (`location_proximity/semantic_clustering.py`)
   - Clusters locations using DBSCAN
   - Generates emotion profiles per cluster

---

## Performance Considerations

### Caching Strategy
- Cache proximity scores between location pairs (TTL: 1 hour)
- Cache cluster results per user (invalidate on new upload)
- Cache hotspot calculations (invalidate on new location-emotion pair)

### Optimization
- Batch proximity calculations for nearby locations
- Precompute distance matrices for frequent queries
- Use spatial indexing (MongoDB geospatial queries) for radius searches

### Expected Performance
- Upload processing: < 3 seconds (including all analysis)
- Proximity query: < 100ms
- Clustering: < 2 seconds for 100 locations
- Dashboard load: < 1 second

---

## Future Enhancements

1. **Real-time Place Enrichment**
   - Google Places API integration for place type inference
   - Automatic tagging of cultural/linguistic attributes

2. **Collaborative Filtering**
   - Cross-user emotion patterns at shared locations
   - Privacy-preserving aggregation

3. **Temporal Predictions**
   - Predict likely emotional response at a location based on history
   - Recommend emotionally beneficial locations

4. **WebSocket Support**
   - Real-time clustering updates
   - Live emotion-location mapping during photo uploads

---

## Testing Strategy

- **Unit Tests**: Mock external dependencies (DB, ML models)
- **Integration Tests**: Test full pipeline with synthetic data
- **Load Tests**: Simulate 100 concurrent users
- **Validation**: Compare results against `tests/data/expected_results.json`

---

**Document Version**: 1.0  
**Last Updated**: February 3, 2026  
**Author**: Krishan (GSoC 2026 Contributor)
