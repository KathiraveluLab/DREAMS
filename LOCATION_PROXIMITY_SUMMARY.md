# Location-Proximity Analysis Extension

## Overview

A new module for DREAMS that analyzes **multi-dimensional location proximity** to understand how semantically similar places influence emotional patterns in recovery journeys.

**Key Innovation**: Goes beyond GPS coordinates to consider categorical, linguistic, and cultural dimensions of location similarity.

---

## Module Location

```
DREAMS/location_proximity/
├── __init__.py
├── location_extractor.py          # Extract GPS from images
├── proximity_calculator.py        # Multi-dimensional proximity
├── emotion_location_mapper.py     # Emotion-location patterns
├── semantic_clustering.py         # Cluster similar places
├── demo.py                        # Demonstration script
├── test_proximity.py              # Test suite
├── requirements.txt               # Dependencies
├── README.md                      # Module documentation
└── RESEARCH.md                    # Research foundation
```

---

## Quick Demo

```bash
cd location_proximity
pip install -r requirements.txt
python demo.py
```

**Output**:
```
DEMO 1: Multi-Dimensional Proximity Calculation
================================================================
St. Mary's Church ↔ Holy Trinity Church : 0.850
Alaska Native Medical Center ↔ Providence Hospital : 0.725

✓ Notice: Two churches have high proximity despite different locations
✓ Notice: Two hospitals cluster together semantically
```

---

## Key Features

### 1. Multi-Dimensional Proximity
Calculates location similarity using:
- **Geographic**: Physical distance (Haversine)
- **Categorical**: Place type (church ↔ church)
- **Linguistic**: Language context
- **Cultural**: Shared cultural tags

### 2. Emotion-Location Mapping
- Track emotional patterns at specific locations
- Identify "emotional hotspots" (places with consistent emotions)
- Compare emotions across place categories
- Temporal emotion trends at locations

### 3. Semantic Clustering
- Group semantically similar places
- Analyze emotional patterns within clusters
- Discover cross-location patterns
- DBSCAN-based clustering (no predefined cluster count)

---

## Research Questions Addressed

1. **Do semantically similar places evoke similar emotions?**
   - Compare two different churches vs. church and hospital

2. **Is categorical proximity more predictive than geographic proximity?**
   - Correlation analysis between proximity dimensions and emotions

3. **Do certain place types consistently associate with specific emotions?**
   - Statistical analysis across place categories

4. **How do place-emotion associations evolve during recovery?**
   - Temporal analysis of location patterns

---

## Example Use Cases

### Use Case 1: Categorical Analysis
**Question**: Do all churches evoke positive emotions, or just specific ones?

```python
from location_proximity.emotion_location_mapper import EmotionLocationMapper

mapper = EmotionLocationMapper()
# Add data...

# Compare specific church vs. all churches
church_a_profile = mapper.get_location_sentiment_profile("church_a")
all_churches = mapper.compare_place_types()["church"]
```

### Use Case 2: Cross-Location Patterns
**Question**: Do Portuguese restaurants evoke similar emotions despite different locations?

```python
from location_proximity.semantic_clustering import find_similar_place_patterns

patterns = find_similar_place_patterns(
    places=portuguese_restaurants,
    emotion_mapper=mapper,
    proximity_threshold=0.6
)
```

### Use Case 3: Recovery Journey Mapping
**Question**: How does a person's emotional relationship with healthcare facilities change over time?

```python
hospitals = [p for p in places if p['type'] == 'hospital']
for hospital in hospitals:
    trend = mapper.temporal_emotion_trend(hospital['id'])
    # Analyze trend...
```

---

## Integration with DREAMS

### Extend Post Schema
```python
# Add to dreamsApp/app/ingestion/routes.py
from location_proximity.location_extractor import extract_location_from_image

location = extract_location_from_image(image_path)
if location:
    post_doc['location'] = {
        'lat': location['lat'],
        'lon': location['lon']
    }
```

### Add Dashboard Route
```python
# Add to dreamsApp/app/dashboard/main.py
@bp.route('/location_analysis/<user_id>')
def location_analysis(user_id):
    # Use EmotionLocationMapper to analyze patterns
    # Render visualization
```

---

## Metrics & Validation

### Clustering Quality
- Silhouette score
- Davies-Bouldin index

### Emotion Prediction
- Accuracy of predicting emotion from place type
- Precision/Recall for hotspot identification

### Statistical Significance
- Chi-square tests for sentiment distributions
- ANOVA across place types
- Effect size calculations

---

## Research Contribution

**Novel Contribution**: First formalization of multi-dimensional location proximity for emotion analysis in recovery journeys.

**Potential Publications**:
1. "Beyond GPS: Multi-Dimensional Location Proximity in Emotional Recovery Analysis"
2. "Semantic Place Similarity and Emotional Patterns in Digitized Memories"
3. "Affective Geography of Recovery: A Computational Approach"

---

## Dependencies

```
Pillow>=10.0.0          # Image EXIF extraction
numpy>=1.24.0           # Numerical computations
scikit-learn>=1.3.0     # Clustering algorithms
```

---

## Testing

```bash
cd location_proximity
pytest test_proximity.py -v
```

**Test Coverage**:
- Geographic distance calculation
- Proximity metrics (all dimensions)
- Emotion-location mapping
- Clustering functionality
- Edge cases and error handling

---

## Future Enhancements

### Phase 1 (Current)
- [x] Multi-dimensional proximity calculation
- [x] Emotion-location mapping
- [x] Semantic clustering
- [x] Demo and tests

### Phase 2 (Next)
- [ ] Google Places API integration
- [ ] Automated place type detection
- [ ] Interactive map visualization (Folium)
- [ ] Real-time clustering

### Phase 3 (Future)
- [ ] Image-based place recognition (CNN)
- [ ] Temporal-spatial modeling
- [ ] Cross-user analysis
- [ ] Causal inference methods

---

## Contributing

This module was developed as part of GSoC 2026 project:
> "Understanding proximity in locations and emotions through digitized memories"

**Mentors**: Jihye Kwon, Pradeeban Kathiravelu  
**Institution**: University of Alaska Fairbanks

Contributions welcome! See [location_proximity/README.md](location_proximity/README.md) for details.

---

## Contact

- **Project**: https://github.com/KathiraveluLab/DREAMS
- **Discussions**: https://github.com/KathiraveluLab/DREAMS/discussions
- **Mentors**: jkwon2@alaska.edu, pkathiravelu@alaska.edu

---

## License

Same as DREAMS project (see [LICENSE](LICENSE))

---

**Status**: ✅ Ready for integration and testing  
**Version**: 0.1.0  
**Last Updated**: 2024
