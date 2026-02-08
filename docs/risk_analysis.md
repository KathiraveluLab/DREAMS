# Risk Analysis & Mitigation Strategies

## Overview

This document identifies potential risks to the DREAMS location-proximity GSoC project and outlines comprehensive mitigation strategies to ensure successful completion within the 350-hour timeframe.

---

## Risk Matrix

| Risk ID | Risk | Probability | Impact | Severity | Mitigation Priority |
|---------|------|-------------|--------|----------|-------------------|
| R1 | EXIF data unavailable in most photos | High | High | **Critical** | 1 |
| R2 | Multi-dimensional proximity doesn't improve results | Medium | High | **High** | 2 |
| R3 | Integration conflicts with existing code | Medium | Medium | **Medium** | 3 |
| R4 | Performance issues with large datasets | Medium | High | **High** | 4 |
| R5 | Scope creep beyond 350 hours | Medium | High | **High** | 5 |
| R6 | MongoDB schema changes break existing features | Low | High | **Medium** | 6 |
| R7 | User study recruitment difficulties | Medium | Low | **Low** | 7 |
| R8 | Clustering produces meaningless results | Low | Medium | **Low** | 8 |
| R9 | Mentor availability constraints | Low | Medium | **Low** | 9 |
| R10 | Technical dependencies (libraries, APIs) fail | Low | Medium | **Low** | 10 |

**Severity Calculation**: Probability × Impact

---

## Detailed Risk Analysis & Mitigation

### R1: EXIF Data Unavailable in Most Photos
**Probability**: High (70%)  
**Impact**: High  
**Description**: Users may upload photos without GPS metadata (social media downloads, privacy-stripped images, scanned photos).

#### Mitigation Strategies

**Primary**: Fallback to manual location entry
```python
if location_data.get('accuracy') == 'none':
    # Prompt user for manual location
    return {"requires_manual_location": True}
```

**Secondary**: Place inference from caption/keywords
```python
# Extract location names from caption
location_mentions = extract_place_names(caption)
# Use geocoding API to get coordinates
coords = geocode(location_mentions[0])
```

**Tertiary**: Use IP-based geolocation as rough estimate
```python
# For logged-in users, approximate from IP
approx_location = geolocate_ip(request.remote_addr)
```

**Validation**: Track percentage of photos with GPS in test data. If < 30%, prioritize fallback mechanisms.

**Timeline Impact**: +5 hours for robust fallback implementation

---

### R2: Multi-Dimensional Proximity Doesn't Improve Results
**Probability**: Medium (40%)  
**Impact**: High  
**Description**: Ablation study may show that additional dimensions (linguistic, cultural) don't significantly improve clustering or emotion prediction over geographic distance alone.

#### Mitigation Strategies

**Primary**: Rigorous ablation study early (Week 8)
- Run all 7 experimental conditions
- If multi-dimensional doesn't outperform, pivot to geo + categorical only
- Document findings as research contribution (negative results are valuable)

**Secondary**: Adaptive weighting
```python
# Learn optimal weights from data
weights = optimize_weights(validation_set)
```

**Tertiary**: Focus on interpretability over performance
- Even if metrics are similar, multi-dimensional may be more interpretable
- User study can validate semantic meaningfulness

**Success Criteria Adjustment**: If multi-dimensional < 5% better than geo+categorical, simplify to two dimensions.

**Timeline Impact**: None (ablation already planned)

---

### R3: Integration Conflicts with Existing Code
**Probability**: Medium (50%)  
**Impact**: Medium  
**Description**: Extending ingestion pipeline and dashboard may conflict with ongoing development or existing functionality.

#### Mitigation Strategies

**Primary**: Regular communication with mentors
- Weekly check-ins on any parallel development
- Review PRs in main branch before integration

**Secondary**: Modular design with clear interfaces
```python
# Use dependency injection for easy testing
class LocationProximityService:
    def __init__(self, db_client, exif_extractor):
        self.db = db_client
        self.exif = exif_extractor
```

**Tertiary**: Feature flags for gradual rollout
```python
if app.config.get('ENABLE_LOCATION_PROXIMITY'):
    # New functionality
```

**Validation**: Integration tests run against latest main branch weekly.

**Timeline Impact**: +10 hours for conflict resolution (already budgeted in Phase 2)

---

### R4: Performance Issues with Large Datasets
**Probability**: Medium (40%)  
**Impact**: High  
**Description**: Proximity calculations for 1000+ locations may exceed 3-second upload target.

#### Mitigation Strategies

**Primary**: Optimization techniques
- **Spatial indexing**: MongoDB geospatial queries for nearby locations
- **Caching**: Cache proximity scores between location pairs
- **Batch processing**: Compute proximity matrix in background task
- **Approximate algorithms**: Use locality-sensitive hashing for large-scale

**Secondary**: Performance benchmarks early
```python
@pytest.mark.benchmark
def test_proximity_performance():
    """Ensure proximity calculation < 100ms for 100 locations."""
    start = time.time()
    compute_proximity_matrix(100_locations)
    assert time.time() - start < 0.1
```

**Tertiary**: Incremental computation
```python
# Only compute proximity for new location vs. existing
# Don't recompute entire matrix on each upload
```

**Success Criteria**: If upload > 3s with 100 locations, move clustering to async background job.

**Timeline Impact**: +8 hours for optimization (included in Week 6-7)

---

### R5: Scope Creep Beyond 350 Hours
**Probability**: Medium (50%)  
**Impact**: High  
**Description**: Feature requests or perfectionism may expand scope beyond planned milestones.

#### Mitigation Strategies

**Primary**: Strict scope management
- **MVP focus**: Core features only (proximity, clustering, hotspots)
- **Future work list**: Document "nice-to-haves" for post-GSoC
- **Weekly hour tracking**: Monitor actual vs. planned hours

**Secondary**: Ruthless prioritization
```
P0 (Must-have): Basic proximity, clustering, integration
P1 (Should-have): Dashboard visualization, ablation study
P2 (Nice-to-have): Advanced analytics, real-time updates
P3 (Future): Cross-user analysis, ML predictions
```

**Tertiary**: Timeboxing
- Each task has maximum hour allocation
- If exceeded, move to "polish" phase or defer

**Validation**: If cumulative hours > planned by 10%, cut P2 features.

**Timeline Impact**: None (proactive management)

---

### R6: MongoDB Schema Changes Break Existing Features
**Probability**: Low (20%)  
**Impact**: High  
**Description**: Adding new collections or fields may inadvertently break existing queries or functionality.

#### Mitigation Strategies

**Primary**: Backward compatibility
```python
# Add fields, don't modify existing ones
post_doc = {
    # ... existing fields unchanged
    'location': location_data  # NEW, optional
}
```

**Secondary**: Comprehensive testing
- Run full existing test suite before/after schema changes
- Integration tests validate old functionality still works

**Tertiary**: Database migrations
```python
# Migration script to add new fields safely
def migrate_add_location_field():
    db.posts.update_many(
        {'location': {'$exists': False}},
        {'$set': {'location': {'accuracy': 'none'}}}
    )
```

**Validation**: All existing tests pass after schema extension.

**Timeline Impact**: +4 hours for careful migration (budgeted)

---

### R7: User Study Recruitment Difficulties
**Probability**: Medium (40%)  
**Impact**: Low  
**Description**: May struggle to recruit 10-15 mental health researchers for user study in August.

#### Mitigation Strategies

**Primary**: Early recruitment
- Start outreach in Week 8 (2 weeks before user study)
- Leverage mentors' professional networks
- Offer small incentive (e.g., $25 Amazon gift card)

**Secondary**: Alternative participants
- PhD students in clinical psychology
- Recovery support group facilitators
- DREAMS/Beehive existing community members

**Tertiary**: Internal validation
- If < 5 external participants, conduct internal review with mentors
- Document as "expert evaluation" instead of "user study"

**Success Criteria**: Minimum 5 participants provides sufficient qualitative feedback.

**Timeline Impact**: None (user study is enhancement, not blocker)

---

### R8: Clustering Produces Meaningless Results
**Probability**: Low (25%)  
**Impact**: Medium  
**Description**: DBSCAN may produce many outliers or fail to find coherent clusters with real data.

#### Mitigation Strategies

**Primary**: Adaptive parameters
```python
# Automatically tune eps and min_samples
from sklearn.model_selection import GridSearchCV
best_params = grid_search_dbscan(proximity_matrix)
```

**Secondary**: Alternative algorithms
- Try HDBSCAN (hierarchical DBSCAN) for adaptive density
- Try Agglomerative Clustering with proximity distance matrix
- Ensemble of multiple clustering methods

**Tertiary**: Fallback to simpler grouping
```python
# If clustering fails, fall back to place-type grouping
if silhouette_score < 0.3:
    # Just group by place_type
    clusters = group_by_place_type(locations)
```

**Validation**: Synthetic dataset should always produce 3 clean clusters.

**Timeline Impact**: +6 hours for parameter tuning (budgeted)

---

### R9: Mentor Availability Constraints
**Probability**: Low (20%)  
**Impact**: Medium  
**Description**: Mentors may have limited availability during summer for weekly meetings.

#### Mitigation Strategies

**Primary**: Asynchronous communication
- Detailed weekly progress reports via email/GitHub discussions
- Use project board (GitHub Projects) for transparency
- Record demo videos for async review

**Secondary**: Flexible meeting schedule
- Schedule meetings 2 weeks in advance
- Offer multiple time slot options
- Accept shorter 30-min check-ins if needed

**Tertiary**: Self-sufficiency
- Make decisions independently when appropriate
- Document rationale for mentor review later
- Escalate only blockers that require immediate input

**Success Criteria**: Minimum 1 mentor interaction per week (meeting or detailed async feedback).

**Timeline Impact**: None

---

### R10: Technical Dependencies Fail
**Probability**: Low (15%)  
**Impact**: Medium  
**Description**: External libraries (scikit-learn, exifread) or services (Google Places API) may have issues.

#### Mitigation Strategies

**Primary**: Pin dependency versions
```txt
# requirements.txt
scikit-learn==1.4.0
exifread==3.0.0
```

**Secondary**: Fallback implementations
```python
# If exifread fails, use Pillow
try:
    from exifread import process_file
except ImportError:
    # Use Pillow fallback
    from PIL import Image
```

**Tertiary**: No external API dependencies for MVP
- Defer Google Places integration to future work
- Core functionality works offline with synthetic place types

**Validation**: Test in clean environment before each phase.

**Timeline Impact**: None (good practice)

---

## Risk Monitoring & Response Plan

### Weekly Risk Review
Every mentor meeting, review:
1. Have any risks materialized?
2. Has probability/impact changed for any risk?
3. Are mitigation strategies working?

### Escalation Criteria
Escalate to mentors immediately if:
- Any critical risk materializes
- Cumulative hours > 10% over plan
- Core functionality blocker arises

### Risk Log
Maintain `docs/risk_log.md` with:
- Date risk identified
- Mitigation actions taken
- Current status
- Lessons learned

---

## Contingency Plans by Phase

### Phase 1 Contingency
**If**: Core implementation takes 140h instead of 120h  
**Then**: Reduce dashboard polish in Phase 2 (cut 20h)

### Phase 2 Contingency
**If**: Integration issues consume extra time  
**Then**: Defer advanced visualizations, focus on basic dashboard

### Phase 3 Contingency
**If**: Evaluation reveals major issues  
**Then**: Allocate Final Week hours to fixes instead of documentation

---

## Success Probability Assessment

Given mitigation strategies:

| Outcome | Probability |
|---------|-------------|
| **Complete Success** (All deliverables, on time) | 70% |
| **Partial Success** (Core features, minor delays) | 25% |
| **Significant Issues** (Major delays or missing features) | 5% |

**Overall Project Risk Level**: **LOW-MEDIUM**

With proactive risk management, rigorous testing, and mentor collaboration, this project has a high likelihood of successful completion within the 350-hour GSoC timeframe.

---

**Version**: 1.0  
**Last Updated**: February 3, 2026  
**Author**: Krishan (GSoC 2026 Contributor)
