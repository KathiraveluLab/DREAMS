# Test Data

Synthetic dataset for location-proximity analysis testing.

## Files

### locations.json
- 9 locations total (3 parks, 3 hospitals, 3 churches)
- All locations in Anchorage, Alaska area
- Each location has coordinates, type, language, cultural tags

### sentiments.json  
- 14 sentiment records across the locations
- Multiple users and timestamps
- Realistic sentiment patterns by place type

### expected_results.json
- Expected proximity scores for validation
- Clustering expectations (3 clusters by type)
- Emotion pattern expectations

## Usage

```python
import json

# Load locations
with open('tests/data/locations.json') as f:
    locations = json.load(f)['locations']

# Load sentiments  
with open('tests/data/sentiments.json') as f:
    sentiments = json.load(f)['sentiment_data']

# Load expected results
with open('tests/data/expected_results.json') as f:
    expected = json.load(f)
```

## Data Characteristics

### Location Distribution
- **Parks**: Recreational, outdoor, nature-focused
- **Hospitals**: Healthcare, medical, emergency services  
- **Churches**: Religious, community, traditional

### Sentiment Patterns
- **Churches**: Predominantly positive (0.75-0.88)
- **Hospitals**: Predominantly negative (0.25-0.35)
- **Parks**: Mixed positive/neutral (0.55-0.92)

### Geographic Spread
- All within ~10km radius in Anchorage
- Realistic coordinate variations
- Suitable for proximity testing