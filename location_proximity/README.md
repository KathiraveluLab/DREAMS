# Location-Proximity Analysis Module

**Understanding proximity in locations and emotions through digitized memories**

This module extends DREAMS to analyze how semantically similar locations (not just geographically close ones) influence emotional patterns in recovery journeys.

---

## Core Concept

Traditional location analysis uses only GPS coordinates. This module introduces **multi-dimensional proximity**:

1. **Geographic Proximity**: Physical distance (Haversine)
2. **Categorical Similarity**: Same place type (church ↔ church)
3. **Linguistic Similarity**: Same language context
4. **Cultural Similarity**: Shared cultural tags

---

## Research Background

### Theoretical Foundation

This module is grounded in **affective geography** - the study of how emotions are spatially distributed and how places influence emotional experiences. Traditional location analysis focuses solely on geographic distance, but recovery narratives reveal that semantically similar locations can evoke similar emotions regardless of physical proximity.

### Key Research Questions

1. **Semantic Similarity**: Can we formalize location proximity beyond GPS coordinates to capture functional, cultural, and linguistic similarities?
2. **Emotion-Place Associations**: Do semantically similar places consistently evoke similar emotions across individuals?
3. **Recovery Applications**: How can understanding place-emotion patterns support personal recovery journeys?

### Multi-Dimensional Proximity Framework

The module implements a composite proximity score combining four dimensions:

- **Geographic Proximity**: Physical distance using Haversine formula
- **Categorical Similarity**: Place type matching (church ↔ church, hospital ↔ clinic)
- **Linguistic Similarity**: Language context and naming patterns
- **Cultural Similarity**: Shared cultural significance and associations

**Composite Score**: P(L₁, L₂) = α·P_geo + β·P_cat + γ·P_ling + δ·P_cult

Where default weights are: α=0.3 (geographic), β=0.4 (categorical), γ=0.15 (linguistic), δ=0.15 (cultural)

### Expected Findings

- Semantically similar places evoke similar emotions across individuals
- Categorical proximity is more predictive of emotional response than geographic proximity alone
- Certain place categories consistently associate with specific emotions
- Emotional associations with place types evolve during recovery trajectories

---

## Related Work

### Affective Geography
- **Thien, D. (2005)**. After or beyond feeling? A consideration of affect and emotion in geography. *Area*, 37(4), 450-454.
- **Davidson, J., & Milligan, C. (2004)**. Embodying emotion sensing space: introducing emotional geographies. *Social & Cultural Geography*, 5(4), 523-542.

### Place and Mental Health
- **Gesler, W. (1992)**. Therapeutic landscapes: medical issues in light of the new cultural geography. *Social Science & Medicine*, 34(7), 735-746.
- **Williams, A. (Ed.). (2007)**. *Therapeutic landscapes*. Routledge.

### Semantic Similarity
- **Ballatore, A., Bertolotto, M., & Wilson, D. C. (2013)**. Computing the semantic similarity of geographic terms using volunteered lexical definitions. In *Proceedings of the 21st ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems* (pp. 414-417).
- **Janowicz, K., Raubal, M., & Kuhn, W. (2011)**. Semantic similarity measurement and geospatial applications. *Transactions in GIS*, 15(3), 299-318.

### Recovery and Place
- **Topor, A., Borg, M., Di Girolamo, S., & Davidson, L. (2011)**. Others: The role of family, friends, and professionals in the recovery process. In *The Routledge international handbook of recovery* (pp. 131-142). Routledge.
- **Borg, M., & Davidson, L. (2008)**. The nature of recovery as lived in everyday experience. *Journal of Mental Health*, 17(2), 129-140.

### Key Insights from Literature

1. **Place Attachment**: Individuals develop emotional bonds with locations that become symbolic in recovery narratives
2. **Therapeutic Landscapes**: Certain environments promote healing and well-being
3. **Semantic Place Similarity**: Locations share meaning beyond physical proximity
4. **Cultural Context**: Place-emotion associations vary by cultural background
5. **Temporal Dynamics**: Emotional responses to places evolve over time

This work extends these foundations by formalizing multi-dimensional proximity for emotion analysis in digitized recovery memories.

---

## Components

### `location_extractor.py`
Extract GPS coordinates from image EXIF data.

```python
from location_extractor import extract_location_from_image

location = extract_location_from_image("photo.jpg")
# Returns: {'lat': 61.2181, 'lon': -149.9003, 'timestamp': '2024:01:15 10:30:00'}
```

### `proximity_calculator.py`
Calculate multi-dimensional proximity between locations.

```python
from proximity_calculator import Place, composite_proximity

place1 = Place("St. Mary's Church", 61.2181, -149.9003, "church", "english")
place2 = Place("Holy Trinity Church", 61.2200, -149.8950, "church", "english")

score = composite_proximity(place1, place2)
# Returns: 0.85 (high proximity despite different locations)
```

### `emotion_location_mapper.py`
Map emotions to locations and discover patterns.

```python
from emotion_location_mapper import EmotionLocationMapper

mapper = EmotionLocationMapper()
mapper.add_entry("church_1", "positive", 0.85, {"place_type": "church"})

profile = mapper.get_location_sentiment_profile("church_1")
hotspots = mapper.find_emotional_hotspots("positive")
comparison = mapper.compare_place_types()
```

### `semantic_clustering.py`
Cluster locations by semantic similarity and emotional patterns.

```python
from semantic_clustering import SemanticLocationClusterer

clusterer = SemanticLocationClusterer(eps=0.3, min_samples=2)
labels = clusterer.cluster_by_proximity(proximity_matrix)
summary = clusterer.cluster_with_emotions(proximity_matrix, emotion_profiles)
```

---

## Quick Start

### Run the Demo

```bash
cd location_proximity
python demo.py
```

This demonstrates:
- Multi-dimensional proximity calculation
- Emotion-location pattern analysis
- Semantic clustering of places

### Example Output

```
DEMO 1: Multi-Dimensional Proximity Calculation
================================================================
St. Mary's Church ↔ Holy Trinity Church : 0.850
St. Mary's Church ↔ Alaska Native Medical Center : 0.120
Holy Trinity Church ↔ Providence Hospital : 0.115
Alaska Native Medical Center ↔ Providence Hospital : 0.725

✓ Notice: Two churches have high proximity despite different locations
✓ Notice: Two hospitals cluster together semantically
```

---

## Research Applications

### 1. Categorical Emotion Analysis
**Question**: Do all churches evoke similar emotions, or just specific ones?

```python
# Find all church visits
church_visits = mapper.get_locations_by_sentiment("positive")
church_profiles = [mapper.get_location_sentiment_profile(loc) for loc in church_visits]

# Compare: specific church vs. church category
```

### 2. Cross-Location Patterns
**Question**: Do semantically similar places evoke similar emotions?

```python
patterns = find_similar_place_patterns(places, emotion_mapper, proximity_threshold=0.6)

for place1, place2, proximity, emotion_comparison in patterns:
    if emotion_comparison['same_emotion']:
        print(f"{place1} and {place2} both evoke {emotion_comparison['place1_dominant']}")
```

### 3. Cultural Proximity Impact
**Question**: Do places with shared cultural context influence emotions similarly?

```python
# Compare Portuguese restaurants vs. other restaurants
weights = {'geographic': 0.1, 'categorical': 0.3, 'linguistic': 0.3, 'cultural': 0.3}
score = composite_proximity(place1, place2, weights=weights)
```

---

## Integration with DREAMS

### Extend Post Schema

```python
# In dreamsApp/app/ingestion/routes.py
from location_proximity.location_extractor import extract_location_from_image
from location_proximity.proximity_calculator import Place

@bp.route('/upload', methods=['POST'])
def upload_post():
    # ... existing code ...
    
    # Extract location
    location = extract_location_from_image(image_path)
    
    if location:
        post_doc['location'] = {
            'lat': location['lat'],
            'lon': location['lon'],
            'place_type': None,  # To be enriched via API
            'language': None
        }
    
    # ... rest of code ...
```

### Add Location Analysis Route

```python
# In dreamsApp/app/dashboard/main.py
from location_proximity.emotion_location_mapper import EmotionLocationMapper

@bp.route('/location_analysis/<user_id>')
def location_analysis(user_id):
    posts = mongo['posts'].find({'user_id': user_id})
    
    mapper = EmotionLocationMapper()
    for post in posts:
        if 'location' in post:
            mapper.add_entry(
                location_id=f"{post['location']['lat']},{post['location']['lon']}",
                sentiment=post['sentiment']['label'],
                score=post['sentiment']['score'],
                metadata={'place_type': post['location'].get('place_type')}
            )
    
    hotspots = mapper.find_emotional_hotspots("positive")
    comparison = mapper.compare_place_types()
    
    return render_template('location_analysis.html', 
                          hotspots=hotspots, 
                          comparison=comparison)
```

---

## Dependencies

```bash
pip install pillow numpy scikit-learn
```

For full DREAMS integration:
```bash
pip install -r ../requirements.txt
```

---

## Future Enhancements

- [ ] Google Places API integration for place enrichment
- [ ] Real-time location clustering as data arrives
- [ ] Interactive map visualization (Folium)
- [ ] Temporal-spatial pattern mining
- [ ] Cross-user location-emotion analysis
- [ ] Cultural proximity formalization (research paper)

---

## Research Contribution

This module addresses the GSoC 2026 project:
> "Understanding proximity in locations and emotions through digitized memories"

**Key Innovation**: Formalizing proximity beyond geo-coordinates to understand:
- Specific place (this church) vs. place category (any church)
- How semantic similarity influences emotional patterns
- Cultural and linguistic dimensions of place-emotion associations

---

## Citation

If you use this module in research, please cite:

```
DREAMS Location-Proximity Analysis Module
KathiraveluLab, University of Alaska Fairbanks
https://github.com/KathiraveluLab/DREAMS
```

---

## Contributing

This is part of GSoC 2026. Contributions welcome!

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## Contact

- Mentors: Jihye Kwon (jkwon2@alaska.edu), Pradeeban Kathiravelu (pkathiravelu@alaska.edu)
- Project: https://github.com/KathiraveluLab/DREAMS
- Discussions: https://github.com/KathiraveluLab/DREAMS/discussions
