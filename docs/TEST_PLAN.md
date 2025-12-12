# Location-Proximity Module Test Plan

## Overview

Comprehensive testing strategy for the location-proximity analysis module, covering unit tests, integration tests, and edge case validation.

## Test Categories

### 1. Unit Tests
- Individual component functionality
- Input validation and error handling
- Mathematical accuracy verification

### 2. Integration Tests
- Component interaction validation
- End-to-end workflow testing
- API endpoint verification

### 3. Performance Tests
- Large dataset processing
- Memory usage optimization
- Response time benchmarks

### 4. Edge Case Tests
- Boundary conditions
- Invalid input handling
- Error recovery scenarios

---

## Location Extractor Tests

### Unit Tests

#### Test Case: LE-UT-001
**Description**: Extract GPS coordinates from valid EXIF data
**Input**: Image with GPS EXIF tags (lat: 64.8378, lon: -147.7164)
**Expected Output**: `{'lat': 64.8378, 'lon': -147.7164, 'altitude': None}`
**Priority**: High

#### Test Case: LE-UT-002
**Description**: Handle image without GPS data
**Input**: Image file with no EXIF GPS information
**Expected Output**: `None`
**Priority**: High

#### Test Case: LE-UT-003
**Description**: Process corrupted image file
**Input**: Corrupted/invalid image file
**Expected Output**: Exception handling, return `None`
**Priority**: Medium

#### Test Case: LE-UT-004
**Description**: Extract GPS with altitude information
**Input**: Image with GPS + altitude EXIF data
**Expected Output**: `{'lat': 64.8378, 'lon': -147.7164, 'altitude': 150.5}`
**Priority**: Low

### Edge Cases

#### Test Case: LE-EC-001
**Description**: GPS coordinates at boundary values
**Input**: Image with lat=90.0, lon=180.0
**Expected Output**: Valid coordinate extraction
**Priority**: Medium

#### Test Case: LE-EC-002
**Description**: Non-existent file path
**Input**: Path to non-existent image file
**Expected Output**: FileNotFoundError handling
**Priority**: High

---

## Proximity Calculator Tests

### Unit Tests

#### Test Case: PC-UT-001
**Description**: Calculate geographic distance (Haversine)
**Input**: 
- Location 1: (64.8378, -147.7164) - Fairbanks
- Location 2: (61.2181, -149.9003) - Anchorage
**Expected Output**: ~358.5 km
**Tolerance**: ±1 km
**Priority**: High

#### Test Case: PC-UT-002
**Description**: Categorical proximity - same type
**Input**: 
- Place 1: {'type': 'church', 'name': 'St. Mary'}
- Place 2: {'type': 'church', 'name': 'Holy Trinity'}
**Expected Output**: 1.0 (perfect match)
**Priority**: High

#### Test Case: PC-UT-003
**Description**: Categorical proximity - related types
**Input**: 
- Place 1: {'type': 'hospital', 'name': 'General Hospital'}
- Place 2: {'type': 'clinic', 'name': 'Health Clinic'}
**Expected Output**: 0.5 (related match)
**Priority**: Medium

#### Test Case: PC-UT-004
**Description**: Categorical proximity - unrelated types
**Input**: 
- Place 1: {'type': 'church', 'name': 'St. Mary'}
- Place 2: {'type': 'restaurant', 'name': 'Pizza Place'}
**Expected Output**: 0.0 (no match)
**Priority**: Medium

#### Test Case: PC-UT-005
**Description**: Linguistic similarity calculation
**Input**: 
- Place 1: {'language': 'portuguese', 'name': 'Casa do Bacalhau'}
- Place 2: {'language': 'portuguese', 'name': 'Restaurante Lisboa'}
**Expected Output**: 1.0 (same language)
**Priority**: Medium

#### Test Case: PC-UT-006
**Description**: Cultural similarity (Jaccard index)
**Input**: 
- Place 1: {'cultural_tags': ['european', 'catholic', 'traditional']}
- Place 2: {'cultural_tags': ['european', 'traditional', 'historic']}
**Expected Output**: 0.5 (2 common out of 4 total unique tags)
**Priority**: Medium

#### Test Case: PC-UT-007
**Description**: Composite proximity calculation
**Input**: Two locations with all proximity dimensions
**Expected Output**: Weighted sum: α·P_geo + β·P_cat + γ·P_ling + δ·P_cult
**Weights**: α=0.3, β=0.4, γ=0.15, δ=0.15
**Priority**: High

### Edge Cases

#### Test Case: PC-EC-001
**Description**: Identical locations
**Input**: Same GPS coordinates and attributes
**Expected Output**: Proximity score = 1.0
**Priority**: High

#### Test Case: PC-EC-002
**Description**: Maximum distance locations
**Input**: Antipodal points (opposite sides of Earth)
**Expected Output**: Geographic proximity ≈ 0.0
**Priority**: Low

#### Test Case: PC-EC-003
**Description**: Missing attribute handling
**Input**: Location with missing 'type' field
**Expected Output**: Graceful degradation, use available dimensions
**Priority**: Medium

---

## Emotion-Location Mapper Tests

### Unit Tests

#### Test Case: ELM-UT-001
**Description**: Add sentiment data for location
**Input**: 
- Location ID: 'loc_001'
- Sentiment: 0.8 (positive)
- Timestamp: '2024-01-15T10:30:00Z'
**Expected Output**: Successfully stored sentiment record
**Priority**: High

#### Test Case: ELM-UT-002
**Description**: Calculate location sentiment profile
**Input**: Location with 5 sentiment records: [0.8, 0.6, 0.9, 0.7, 0.5]
**Expected Output**: 
- Mean: 0.7
- Std: 0.15
- Count: 5
**Priority**: High

#### Test Case: ELM-UT-003
**Description**: Identify emotional hotspot
**Input**: Location with consistent positive sentiment (≥60% above 0.6)
**Expected Output**: Classified as positive hotspot
**Priority**: Medium

#### Test Case: ELM-UT-004
**Description**: Compare place types sentiment
**Input**: Multiple churches and hospitals with sentiment data
**Expected Output**: Mean sentiment for 'church' (e.g., 0.75) is significantly higher than for 'hospital' (e.g., 0.4), with a t-test p-value < 0.05
**Priority**: Medium

#### Test Case: ELM-UT-005
**Description**: Temporal emotion trend analysis
**Input**: Location with sentiment data over 6 months
**Expected Output**: Trend is 'improving' if linear regression slope > 0.1, 'declining' if slope < -0.1, 'stable' if -0.1 ≤ slope ≤ 0.1
**Priority**: Low

### Edge Cases

#### Test Case: ELM-EC-001
**Description**: Location with single sentiment record
**Input**: One sentiment value for location
**Expected Output**: Profile with count=1, std=0
**Priority**: Medium

#### Test Case: ELM-EC-002
**Description**: Location with no sentiment data
**Input**: Request profile for location without data
**Expected Output**: Empty profile or appropriate default
**Priority**: High

---

## Semantic Clustering Tests

### Unit Tests

#### Test Case: SC-UT-001
**Description**: DBSCAN clustering with optimal parameters
**Input**: 
- 15 locations (5 churches, 5 hospitals, 5 restaurants)
- eps=0.4, min_samples=2
**Expected Output**: 3 distinct clusters
**Priority**: High

#### Test Case: SC-UT-002
**Description**: Cluster quality metrics calculation
**Input**: Generated clusters from test dataset
**Expected Output**: 
- Silhouette score > 0.5
- Davies-Bouldin index < 1.0
**Priority**: Medium

#### Test Case: SC-UT-003
**Description**: Find similar place patterns
**Input**: 
- Portuguese restaurants across different locations
- Proximity threshold: 0.6
**Expected Output**: Grouped similar restaurants
**Priority**: Medium

#### Test Case: SC-UT-004
**Description**: Noise point identification
**Input**: Dataset with outlier locations
**Expected Output**: Outliers labeled as noise (-1)
**Priority**: Low

### Edge Cases

#### Test Case: SC-EC-001
**Description**: Insufficient data for clustering
**Input**: Dataset with only 2 locations
**Expected Output**: Both points are labeled as noise (cluster label -1) if their distance exceeds eps. Otherwise, they form a single cluster.
**Priority**: Medium

#### Test Case: SC-EC-002
**Description**: All locations identical
**Input**: Multiple locations with identical attributes
**Expected Output**: Single cluster containing all points
**Priority**: Low

---

## Integration Tests

### Test Case: INT-001
**Description**: End-to-end proximity analysis workflow
**Steps**:
1. Extract GPS from test images
2. Calculate multi-dimensional proximity
3. Map emotions to locations
4. Perform semantic clustering
**Expected Output**: Complete analysis pipeline execution
**Priority**: High

### Test Case: INT-002
**Description**: API endpoint integration
**Steps**:
1. POST /api/locations/analyze with image data
2. Verify proximity calculations
3. Check emotion mapping results
**Expected Output**: Valid JSON response with analysis results
**Priority**: High

### Test Case: INT-003
**Description**: Database integration
**Steps**:
1. Store location and sentiment data
2. Retrieve for analysis
3. Update with new calculations
**Expected Output**: Persistent data storage and retrieval
**Priority**: Medium

---

## Performance Tests

### Test Case: PERF-001
**Description**: Large dataset processing
**Input**: 1000 locations with full attribute data
**Expected Output**: Processing time < 30 seconds
**Priority**: Medium

### Test Case: PERF-002
**Description**: Memory usage optimization
**Input**: Batch processing of 500 images
**Expected Output**: Memory usage < 1GB peak
**Priority**: Low

### Test Case: PERF-003
**Description**: Concurrent request handling
**Input**: 10 simultaneous proximity calculations
**Expected Output**: All requests complete successfully
**Priority**: Low

---

## Test Data Requirements

### Synthetic Test Dataset
```
tests/data/
├── images/
│   ├── church_with_gps.jpg
│   ├── hospital_no_gps.jpg
│   ├── restaurant_corrupted.jpg
│   └── park_with_altitude.jpg
├── locations.json
├── sentiments.csv
└── expected_results.json
```

### Location Test Data Structure
```json
{
  "locations": [
    {
      "id": "loc_001",
      "name": "St. Mary's Church",
      "type": "church",
      "coordinates": {"lat": 64.8378, "lon": -147.7164},
      "language": "english",
      "cultural_tags": ["christian", "traditional", "community"]
    }
  ]
}
```

### Sentiment Test Data Structure
```csv
location_id,sentiment_score,timestamp,user_id
loc_001,0.8,2024-01-15T10:30:00Z,user_123
loc_001,0.6,2024-01-16T14:20:00Z,user_123
```

---

## Test Execution Strategy

### Automated Testing
- **Unit Tests**: Run on every commit (pytest)
- **Integration Tests**: Run on pull requests
- **Performance Tests**: Run weekly on main branch

### Manual Testing
- **Edge Case Validation**: Monthly review
- **User Acceptance Testing**: Before major releases
- **Security Testing**: Quarterly assessment

### Continuous Integration
```yaml
# .github/workflows/test.yml
- name: Run Location-Proximity Tests
  run: |
    cd location_proximity
    pytest test_proximity.py -v --cov=.
    pytest ../tests/integration/ -v
```

---

## Success Criteria

### Unit Test Coverage
- **Minimum**: 85% code coverage
- **Target**: 95% code coverage
- **Critical paths**: 100% coverage

### Performance Benchmarks
- **Proximity calculation**: < 10ms per pair
- **Clustering**: < 5 seconds for 100 locations
- **Memory usage**: < 100MB for standard operations

### Quality Metrics
- **All tests pass**: Zero failing tests in CI
- **Code quality**: Pylint score > 8.0
- **Documentation**: All public functions documented

---

## Test Maintenance

### Regular Updates
- Update test data quarterly
- Review edge cases after bug reports
- Benchmark performance monthly

### Test Data Management
- Version control test datasets
- Anonymize real-world test data
- Maintain data consistency across environments