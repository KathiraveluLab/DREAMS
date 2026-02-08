# Evaluation Metrics & Ablation Study Plan

## Overview

This document defines the evaluation methodology for the DREAMS location-proximity module, including quantitative metrics, qualitative assessment criteria, and a systematic ablation study plan to validate the multi-dimensional proximity approach.

---

## 1. Quantitative Evaluation Metrics

### 1.1 Proximity Calculation Accuracy

**Metric**: Mean Absolute Error (MAE) against human-annotated proximity scores

**Method**:
- Collect human judgments for 50 location pairs
- Humans rate semantic similarity on 0-1 scale
- Compare with computed proximity scores

**Formula**:
```
MAE = (1/n) * Σ|human_score_i - computed_score_i|
```

**Success Criteria**: MAE < 0.15

### 1.2 Clustering Quality

**Metrics**:
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to +1)
- **Davies-Bouldin Index**: Lower is better (minimum 0)
- **Purity**: Percentage of correctly clustered items

**Success Criteria**:
- Silhouette Score > 0.5
- Davies-Bouldin Index < 1.0
- Purity > 0.80

**Validation**:
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Using synthetic data from tests/data/expected_results.json
expected_clusters = {
    0: ["church_001", "church_002", "church_003"],
    1: ["hospital_001", "hospital_002", "hospital_003"],
    2: ["park_001", "park_002", "park_003"]
}

# Compute metrics
silhouette = silhouette_score(proximity_matrix, cluster_labels)
davies_bouldin = davies_bouldin_score(proximity_matrix, cluster_labels)
purity = compute_purity(cluster_labels, ground_truth_labels)
```

### 1.3 Hotspot Detection Precision & Recall

**Metrics**:
- **Precision**: Of detected hotspots, how many are true positives?
- **Recall**: Of true hotspots, how many were detected?
- **F1 Score**: Harmonic mean of precision and recall

**Ground Truth**: Manually labeled hotspots in test dataset

**Formula**:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Success Criteria**: F1 Score > 0.75

### 1.4 Emotion Prediction Accuracy

**Metric**: Accuracy of predicting emotion at a location based on place type

**Method**:
- Hold out 20% of emotion-location pairs
- Predict sentiment using place-type averages
- Compare with ground truth

**Formula**:
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Success Criteria**: Accuracy > 0.65 (better than random baseline of 0.33)

### 1.5 Performance Benchmarks

**Metrics**:
- Processing time per photo upload
- Proximity calculation latency
- Clustering computation time
- Memory usage

**Success Criteria**:
- Upload processing: < 3 seconds
- Proximity query: < 100ms
- Clustering (100 locations): < 2 seconds
- Memory footprint: < 500MB

---

## 2. Qualitative Evaluation

### 2.1 User Study Design

**Participants**: 10-15 mental health researchers and clinicians

**Tasks**:
1. Review 5 user recovery timelines with location-emotion visualizations
2. Assess whether location clusters match their clinical intuition
3. Evaluate usefulness of hotspot identification
4. Rate interpretability of proximity scores (1-5 scale)

**Questions**:
- "Do the location clusters make semantic sense?"
- "Are emotional hotspots clinically meaningful?"
- "Would this analysis support recovery tracking?"
- "Are proximity scores interpretable?"

**Success Criteria**: 
- Mean usefulness rating > 3.5/5
- 70%+ agreement on cluster meaningfulness

### 2.2 Case Study Analysis

**Method**: Detailed analysis of 3 synthetic user journeys

**Dimensions**:
- Temporal evolution of place-emotion associations
- Identification of recovery milestones via location patterns
- Discovery of unexpected semantic proximity patterns

**Documentation**: Rich narrative descriptions with visualizations

---

## 3. Ablation Study Plan

### 3.1 Study Overview

**Purpose**: Determine the contribution of each proximity dimension (geographic, categorical, linguistic, cultural) to overall system performance.

**Method**: Systematically remove each dimension and measure impact on clustering quality and emotion prediction accuracy.

### 3.2 Experimental Conditions

| Condition | Geographic | Categorical | Linguistic | Cultural | Description |
|-----------|------------|-------------|------------|----------|-------------|
| **Full**  | Yes | Yes | Yes | Yes | All dimensions (baseline) |
| **Ablate-Geo** | No | Yes | Yes | Yes | Remove geographic distance |
| **Ablate-Cat** | Yes | No | Yes | Yes | Remove categorical similarity |
| **Ablate-Ling** | Yes | Yes | No | Yes | Remove linguistic context |
| **Ablate-Cult** | Yes | Yes | Yes | No | Remove cultural tags |
| **Geo-Only** | Yes | No | No | No | Geographic distance only |
| **Cat-Only** | No | Yes | No | No | Categorical similarity only |

### 3.3 Evaluation for Each Condition

**Metrics Measured**:
- Silhouette Score
- Davies-Bouldin Index
- Clustering Purity
- Emotion Prediction Accuracy
- Human Interpretability Rating (qualitative)

**Dataset**: `tests/data/locations.json` with 17 locations across 7 types

### 3.4 Expected Outcomes

**Hypothesis 1**: Categorical dimension contributes most to clustering quality
- **Rationale**: Place type (church, hospital) is strongest semantic signal
- **Test**: Ablate-Cat should show largest performance drop

**Hypothesis 2**: Geographic dimension alone is insufficient
- **Rationale**: Two distant churches are more similar than a church and nearby hospital
- **Test**: Geo-Only should have poor clustering purity

**Hypothesis 3**: Multi-dimensional approach outperforms single dimensions
- **Rationale**: Combined signals capture richer semantics
- **Test**: Full model should achieve best metrics

### 3.5 Implementation

```python
# ablation_study.py

import json
import numpy as np
from sklearn.metrics import silhouette_score
from location_proximity.proximity_calculator import composite_proximity

def run_ablation_study():
    """Run systematic ablation study on proximity dimensions."""
    
    # Load test data
    with open('tests/data/locations.json') as f:
        locations = json.load(f)['locations']
    
    # Define ablation conditions
    conditions = {
        'Full': {'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15},
        'Ablate-Geo': {'geo': 0.0, 'cat': 0.55, 'ling': 0.225, 'cult': 0.225},
        'Ablate-Cat': {'geo': 0.5, 'cat': 0.0, 'ling': 0.25, 'cult': 0.25},
        'Ablate-Ling': {'geo': 0.35, 'cat': 0.47, 'ling': 0.0, 'cult': 0.18},
        'Ablate-Cult': {'geo': 0.35, 'cat': 0.47, 'ling': 0.18, 'cult': 0.0},
        'Geo-Only': {'geo': 1.0, 'cat': 0.0, 'ling': 0.0, 'cult': 0.0},
        'Cat-Only': {'geo': 0.0, 'cat': 1.0, 'ling': 0.0, 'cult': 0.0}
    }
    
    results = {}
    
    for condition_name, weights in conditions.items():
        # Compute proximity matrix with current weights
        proximity_matrix = compute_proximity_matrix(locations, weights)
        
        # Cluster using DBSCAN
        from location_proximity.semantic_clustering import SemanticLocationClusterer
        clusterer = SemanticLocationClusterer(eps=0.4, min_samples=2)
        labels = clusterer.cluster_by_proximity(proximity_matrix)
        
        # Compute metrics
        silhouette = silhouette_score(proximity_matrix, labels) if len(set(labels)) > 1 else 0
        purity = compute_purity(labels, ground_truth_from_place_types(locations))
        
        results[condition_name] = {
            'silhouette': silhouette,
            'purity': purity,
            'num_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }
    
    return results
```

### 3.6 Results Documentation

Results will be documented in:
- **Quantitative Table**: Metrics for each condition
- **Visualization**: Bar charts comparing conditions
- **Statistical Analysis**: ANOVA to test significance of differences
- **Interpretation**: Narrative explanation of findings

---

## 4. Validation Against Expected Results

### 4.1 Synthetic Dataset Validation

**File**: `tests/data/expected_results.json`

**Tests**:
1. **Proximity Scores**: Verify computed scores fall within expected ranges
```python
# Church-Church proximity should be 0.8-1.0
assert 0.8 <= compute_proximity(church_001, church_002) <= 1.0

# Church-Hospital proximity should be 0.1-0.4
assert 0.1 <= compute_proximity(church_001, hospital_001) <= 0.4
```

2. **Clustering**: Verify 3 clusters detected (parks, hospitals, churches)
```python
assert num_clusters == 3
assert set(clusters[0]) == set(["church_001", "church_002", "church_003"])
```

3. **Emotion Patterns**: Verify place-type emotion distributions
```python
church_sentiment = aggregate_by_place_type('church')
assert church_sentiment['positive'] >= 0.70  # Expected mean 0.75
```

### 4.2 Test Suite

All validation tests in `tests/test_evaluation_metrics.py`:

```python
def test_proximity_accuracy():
    """Test proximity scores against expected ranges."""
    # Implementation

def test_clustering_quality():
    """Test clustering meets quality thresholds."""
    # Implementation

def test_hotspot_detection():
    """Test hotspot detection precision/recall."""
    # Implementation

def test_emotion_prediction():
    """Test emotion prediction accuracy."""
    # Implementation

def test_performance_benchmarks():
    """Test processing times meet requirements."""
    # Implementation
```

---

## 5. Baseline Comparisons

### 5.1 Baseline Methods

**Baseline 1: Geographic Distance Only**
- Use Haversine formula only
- No semantic considerations

**Baseline 2: K-Means Clustering (Fixed K=3)**
- Traditional clustering without proximity matrix
- Geographic features only

**Baseline 3: Random Emotion Prediction**
- Predict emotions randomly (33% each class)
- Lower bound on performance

### 5.2 Comparison Metrics

| Metric | Random | Geo-Only | K-Means | Multi-Dim (Ours) |
|--------|--------|----------|---------|------------------|
| Silhouette Score | - | TBD | TBD | **Target: > 0.5** |
| Clustering Purity | 33% | TBD | TBD | **Target: > 80%** |
| Emotion Prediction | 33% | TBD | TBD | **Target: > 65%** |
| Interpretability | Low | Medium | Low | **Target: High** |

---

## 6. Continuous Monitoring

### 6.1 Production Metrics

Once deployed, monitor:
- Average proximity calculation time
- Clustering success rate (% of users with valid clusters)
- User engagement with location analysis dashboard
- Error rates in EXIF extraction

### 6.2 A/B Testing

**Test**: Multi-dimensional proximity vs. Geographic-only

**Metrics**:
- Dashboard engagement time
- User-reported usefulness
- Clinical insights discovered

**Duration**: 4 weeks with 50 users per group

---

## 7. Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Metric Implementation** | Week 1 | All metrics coded and tested |
| **Ablation Study** | Week 2 | Results for all conditions |
| **User Study** | Week 3-4 | Qualitative feedback collected |
| **Baseline Comparison** | Week 2 | Comparison table completed |
| **Documentation** | Week 5 | Final evaluation report |

---

## 8. Success Criteria Summary

- **Proximity Accuracy**: MAE < 0.15  
- **Clustering Quality**: Silhouette > 0.5, Purity > 0.80  
- **Hotspot Detection**: F1 > 0.75  
- **Emotion Prediction**: Accuracy > 0.65  
- **Performance**: Upload < 3s, Query < 100ms  
- **User Study**: Usefulness > 3.5/5  
- **Ablation Study**: Multi-dimensional > single dimensions  

---

**Version**: 1.0  
**Last Updated**: February 3, 2026  
**Author**: Krishan (GSoC 2026 Contributor)
