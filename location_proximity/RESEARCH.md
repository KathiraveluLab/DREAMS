# Research Foundation: Location-Emotion Proximity Analysis

## Abstract

This document outlines the theoretical and methodological foundation for analyzing proximity in locations beyond geographic coordinates, specifically in the context of emotional recovery journeys documented through digitized memories.

---

## 1. Research Problem

### 1.1 Traditional Limitation
Existing location-based emotion analysis relies primarily on GPS coordinates, treating proximity as purely geographic distance. This approach fails to capture:

- **Semantic similarity**: Two churches may be far apart but functionally similar
- **Cultural context**: Places sharing cultural significance
- **Linguistic associations**: Language-specific place meanings
- **Categorical patterns**: How place types (hospitals, parks) influence emotions

### 1.2 Research Question
**"How do semantically similar locations (not just geographically proximate ones) influence emotional patterns in recovery journeys?"**

Sub-questions:
1. Can we formalize multi-dimensional proximity for location analysis?
2. Do semantically similar places evoke similar emotions across individuals?
3. What role do categorical, linguistic, and cultural dimensions play?
4. How does this understanding improve recovery journey analysis?

---

## 2. Theoretical Framework

### 2.1 Affective Geography
**Definition**: The study of how emotions are spatially distributed and how places influence emotional experiences.

**Key Concepts**:
- **Place attachment**: Emotional bonds between people and places
- **Emotional topography**: Spatial distribution of emotional experiences
- **Therapeutic landscapes**: Places that promote healing and well-being

**Application to DREAMS**:
- Recovery journeys are inherently spatial
- Certain place types may consistently evoke specific emotions
- Understanding place-emotion associations aids intervention design

### 2.2 Semantic Place Similarity
**Definition**: Similarity between locations based on meaning, function, and context rather than physical distance.

**Dimensions**:
1. **Categorical**: Place type (church, hospital, park)
2. **Functional**: Purpose served (worship, healthcare, recreation)
3. **Linguistic**: Language context and naming
4. **Cultural**: Shared cultural significance

**Relevance**:
- Enables analysis of "all churches" vs. "this specific church"
- Identifies patterns across place categories
- Supports generalization of findings

### 2.3 Recovery Narratives and Place
**Context**: Individuals in recovery often document their journey through photos and stories tied to specific locations.

**Observations**:
- Certain places become symbolic in recovery narratives
- Recurring visits to similar place types may indicate patterns
- Emotional associations with places evolve over time

---

## 3. Methodology

### 3.1 Multi-Dimensional Proximity Formalization

**Composite Proximity Score**:
```
P(L₁, L₂) = α·P_geo(L₁, L₂) + β·P_cat(L₁, L₂) + γ·P_ling(L₁, L₂) + δ·P_cult(L₁, L₂)
```

Where:
- **P_geo**: Geographic proximity (Haversine distance, normalized)
- **P_cat**: Categorical similarity (place type matching)
- **P_ling**: Linguistic similarity (language context)
- **P_cult**: Cultural similarity (Jaccard index of cultural tags)
- **α, β, γ, δ**: Weights (Σ = 1.0)

**Weight Selection**:
- Default: α=0.3, β=0.4, γ=0.15, δ=0.15
- Emphasizes categorical similarity (place type)
- Adjustable based on research focus

### 3.2 Categorical Similarity Calculation

**Exact Match**: Same place type → 1.0
```
P_cat(church, church) = 1.0
```

**Related Categories**: Functionally similar → 0.5
```
P_cat(hospital, clinic) = 0.5
```

**Different Categories**: Unrelated → 0.0
```
P_cat(church, hospital) = 0.0
```

**Category Groups**:
- Religious: {church, temple, mosque, synagogue}
- Healthcare: {hospital, clinic, pharmacy}
- Recreation: {park, garden, beach}
- Food & Drink: {restaurant, cafe, bar}

### 3.3 Emotion-Location Association Mining

**Data Structure**:
```
EmotionLocationEntry = {
    location_id: str,
    sentiment: {'positive', 'negative', 'neutral'},
    score: float [0-1],
    timestamp: datetime,
    place_type: str,
    user_id: str
}
```

**Analysis Methods**:

1. **Location Sentiment Profile**:
   - Aggregate all visits to a location
   - Calculate sentiment distribution
   - Identify dominant emotion

2. **Emotional Hotspots**:
   - Locations with consistent emotional associations
   - Threshold: ≥60% of visits share same sentiment
   - Minimum visits: 3

3. **Place Type Comparison**:
   - Aggregate emotions by place category
   - Calculate percentage distributions
   - Identify category-emotion patterns

### 3.4 Semantic Clustering

**Algorithm**: DBSCAN (Density-Based Spatial Clustering)

**Rationale**:
- No need to predefine cluster count
- Handles noise (outlier locations)
- Works with proximity matrix

**Parameters**:
- **eps**: Maximum distance for neighborhood (0.3-0.5)
- **min_samples**: Minimum points for core point (2-3)

**Process**:
1. Calculate proximity matrix for all locations
2. Convert proximity to distance: D = 1 - P
3. Apply DBSCAN with precomputed distance matrix
4. Analyze emotional patterns within clusters

---

## 4. Expected Findings

### 4.1 Hypotheses

**H1**: Semantically similar places evoke similar emotions across individuals
- **Test**: Compare emotion distributions for places in same cluster
- **Metric**: Chi-square test for sentiment distribution similarity

**H2**: Categorical proximity is more predictive of emotional response than geographic proximity
- **Test**: Correlation analysis between proximity dimensions and emotion similarity
- **Metric**: Pearson correlation coefficient

**H3**: Certain place categories consistently associate with specific emotions
- **Test**: ANOVA across place types for sentiment scores
- **Metric**: F-statistic and effect size

**H4**: Emotional associations with place types evolve during recovery
- **Test**: Temporal analysis of place-emotion patterns
- **Metric**: Trend analysis and change point detection

### 4.2 Validation Metrics

1. **Clustering Quality**:
   - Silhouette score
   - Davies-Bouldin index

2. **Emotion Prediction**:
   - Accuracy of predicting emotion from place type
   - Precision/Recall for emotional hotspot identification

3. **Pattern Significance**:
   - Statistical significance (p < 0.05)
   - Effect size (Cohen's d)

---

## 5. Applications

### 5.1 Clinical Applications
- **Personalized Interventions**: Identify therapeutic locations for individuals
- **Trigger Identification**: Detect places associated with negative emotions
- **Progress Tracking**: Monitor emotional evolution across place types

### 5.2 Research Applications
- **Recovery Patterns**: Understand common place-emotion trajectories
- **Cultural Factors**: Analyze how cultural context influences place-emotion associations
- **Comparative Studies**: Cross-population analysis of location-emotion patterns

### 5.3 System Design
- **Recommendation Systems**: Suggest beneficial locations based on patterns
- **Alert Systems**: Warn about potentially triggering locations
- **Visualization**: Interactive maps showing emotional topography

---

## 6. Limitations and Future Work

### 6.1 Current Limitations
- Requires GPS-tagged images
- Place type classification needs manual annotation or API
- Cultural tags require domain knowledge
- Limited to documented locations (selection bias)

### 6.2 Future Directions
1. **Automated Place Enrichment**: Google Places API integration
2. **Image-Based Place Recognition**: CNN for place type classification
3. **Temporal-Spatial Modeling**: Time-series analysis of location patterns
4. **Cross-User Analysis**: Population-level place-emotion patterns
5. **Causal Inference**: Establish causality between place and emotion

---

## 7. Literature Review (Key Papers)

### Affective Geography
- Thien, D. (2005). "After or beyond feeling? A consideration of affect and emotion in geography"
- Davidson, J., & Milligan, C. (2004). "Embodying emotion sensing space: introducing emotional geographies"

### Place and Mental Health
- Gesler, W. (1992). "Therapeutic landscapes: medical issues in light of the new cultural geography"
- Williams, A. (2007). "Therapeutic landscapes"

### Semantic Similarity
- Ballatore, A., et al. (2013). "Computing the semantic similarity of geographic terms using volunteered lexical definitions"
- Janowicz, K., et al. (2011). "Semantic similarity measurement and geospatial applications"

### Recovery and Place
- Topor, A., et al. (2011). "Others: The role of family, friends, and professionals in the recovery process"
- Borg, M., & Davidson, L. (2008). "The nature of recovery as lived in everyday experience"

---

## 8. Contribution to DREAMS

This module enables DREAMS to:

1. **Formalize proximity** beyond GPS coordinates
2. **Discover patterns** across semantically similar locations
3. **Understand context** of emotional experiences
4. **Support research** on place-emotion associations in recovery
5. **Enable interventions** based on location-emotion insights

**Novel Contribution**: First formalization of multi-dimensional location proximity for emotion analysis in recovery journeys.

---

## 9. Evaluation Plan

### 9.1 Quantitative Evaluation
- Clustering quality metrics
- Emotion prediction accuracy
- Statistical significance tests

### 9.2 Qualitative Evaluation
- Case studies of individual recovery journeys
- Expert review by clinicians
- User feedback on insights

### 9.3 Validation Dataset
- Minimum 50 users
- Minimum 500 location-tagged posts
- Diverse place types and emotions
- Temporal span: 3-6 months

---

## 10. Ethical Considerations

- **Privacy**: Location data is sensitive; anonymization required
- **Consent**: Users must consent to location tracking
- **Bias**: Avoid stigmatizing certain locations or communities
- **Transparency**: Clear communication about how location data is used
- **Beneficence**: Ensure findings benefit recovery support

---

## References

[To be populated with full citations]

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Authors**: DREAMS Research Team, KathiraveluLab
