# Location-Proximity Analysis Module

**Understanding proximity in locations and emotions through digitized memories**

This module extends DREAMS to analyze how semantically similar locations (not just geographically close ones) influence emotional patterns in recovery journeys.

---

## 🎯 Core Concept

Traditional location analysis uses only GPS coordinates. This module introduces **multi-dimensional proximity**:

1. **Geographic Proximity**: Physical distance (Haversine)
2. **Categorical Similarity**: Same place type (church ↔ church)
3. **Linguistic Similarity**: Same language context
4. **Cultural Similarity**: Shared cultural tags

---

## 📦 Components

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

## 🚀 Quick Start

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

## 🔬 Research Applications

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

## 📊 Integration with DREAMS

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

## 🛠️ Dependencies

```bash
pip install pillow numpy scikit-learn
```

For full DREAMS integration:
```bash
pip install -r ../requirements.txt
```

---

## 📈 Future Enhancements

- [ ] Google Places API integration for place enrichment
- [ ] Real-time location clustering as data arrives
- [ ] Interactive map visualization (Folium)
- [ ] Temporal-spatial pattern mining
- [ ] Cross-user location-emotion analysis
- [ ] Cultural proximity formalization (research paper)

---

## 🎓 Research Contribution

This module addresses the GSoC 2026 project:
> "Understanding proximity in locations and emotions through digitized memories"

**Key Innovation**: Formalizing proximity beyond geo-coordinates to understand:
- Specific place (this church) vs. place category (any church)
- How semantic similarity influences emotional patterns
- Cultural and linguistic dimensions of place-emotion associations

---

## 📝 Citation

If you use this module in research, please cite:

```
DREAMS Location-Proximity Analysis Module
KathiraveluLab, University of Alaska Fairbanks
https://github.com/KathiraveluLab/DREAMS
```

---

## 🤝 Contributing

This is part of GSoC 2026. Contributions welcome!

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## 📧 Contact

- Mentors: Jihye Kwon (jkwon2@alaska.edu), Pradeeban Kathiravelu (pkathiravelu@alaska.edu)
- Project: https://github.com/KathiraveluLab/DREAMS
- Discussions: https://github.com/KathiraveluLab/DREAMS/discussions
