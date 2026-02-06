# DREAMS Project Roadmap & Future Work

## Overview

This document outlines the current state of the DREAMS platform, the location-proximity module implementation roadmap for GSoC 2026, and future enhancements for continued development post-GSoC.

---

## Current State (February 2026)

### Completed Features

**Core DREAMS Platform**:
- Flask backend with user authentication
- Image upload and caption processing
- BLIP-based image captioning
- RoBERTa sentiment analysis
- Keyword extraction and HDBSCAN clustering
- MongoDB storage for posts, keywords, and themes
- LLM-based thematic analysis (Gemini integration)
- Dashboard with sentiment timelines and word clouds

**Location-Proximity Foundation** (Pre-GSoC):
- Comprehensive research documentation
- System architecture diagrams
- Integration with existing EXIF extractor (PR #77 by kunal-595)
- Integration with emotion-timeline proximity (PR #70 by AnvayKharb)
- Test plan and evaluation framework
- Synthetic dataset with 17 locations and expected results
- Function stubs and interface definitions for multi-dimensional proximity
- API design specification
- Data model extensions

### In Progress

- Implementation of 6 basic location functions (building on PR #77)
- Multi-dimensional proximity calculator (geographic + categorical + linguistic + cultural)
- Emotion-location mapper (integrating with PR #70's emotion proximity)
- Semantic clustering module (complementing existing emotion timeline segmentation)

---


## GSoC 2026 Roadmap (Aligned with Official Timeline)



### Pre-GSoC: Community Bonding & Planning (Feb 19 – Apr 30, 2026)
**Duration**: 10 weeks | **Effort**: ~40h (prep, onboarding, planning; not counted in GSoC 350h coding period)

- Finalize project requirements and architecture
- Deep-dive into DREAMS codebase and data models
- Refine test plans and synthetic datasets
- Mentor meetings and onboarding


### Phase 1: Core Implementation (May 1 – July 7, 2026)
**Duration**: 10 weeks | **Effort**: 150h

#### Deliverables:
1. **Basic Location Functions** (May)
   - `calculate_distance()` - Haversine formula
   - `validate_coordinates()` - GPS validation
   - `extract_location()` - EXIF integration
   - `compute_proximity()` - Distance + threshold check
   - `find_nearby_locations()` - Radius-based search
   - `cluster_locations()` - Simple geographic clustering

2. **Multi-Dimensional Proximity** (June)
   - `Place` class with categorical/linguistic/cultural attributes
   - Geographic proximity (normalized Haversine)
   - Categorical similarity (place type matching)
   - Linguistic similarity (language context)
   - Cultural similarity (Jaccard index on tags)
   - Composite proximity with configurable weights

3. **Emotion-Location Integration** (late June – early July)
   - `EmotionLocationMapper` class
   - `add_entry()` - Store emotion-location pairs
   - `get_location_sentiment_profile()` - Aggregate per location
   - `find_emotional_hotspots()` - Detect consistent emotions
   - `compare_place_types()` - Category-level patterns
   - `temporal_emotion_trend()` - Time-series analysis

4. **Semantic Clustering** (July)
   - `SemanticLocationClusterer` with DBSCAN
   - Cluster emotion profile aggregation
   - Visualization support
   - Parameter tuning utilities

**Milestones**:
- All unit tests passing (90%+ coverage)
- Validated against synthetic dataset
- Performance benchmarks met

---


### Phase 2: Integration & Testing (July 8 – September 1, 2026)
**Duration**: 8 weeks | **Effort**: 110h

#### Deliverables:
1. **Backend Integration** (July)
   - Extend `app/ingestion/routes.py` with location extraction
   - Implement 4 REST API endpoints:
     - `POST /api/upload` (enhanced with location)
     - `GET /api/location/proximity`
     - `POST /api/location/cluster`
     - `GET /api/location/hotspots`
   - MongoDB schema extensions and indexes
   - API authentication and rate limiting

2. **Dashboard Visualization** (August)
   - `/location_analysis/<user_id>` route
   - HTML/CSS template with:
     - Interactive map (Leaflet.js) with hotspot markers
     - Cluster visualization cards
     - Place-type comparison bar charts
     - Temporal emotion patterns
   - JavaScript for dynamic content loading
   - Mobile-responsive design

3. **End-to-End Testing** (August)
   - Integration tests for full pipeline
   - Performance testing (upload < 3s, clustering < 2s)
   - Load testing (100 concurrent users)
   - Cross-browser compatibility testing

**Milestones**:
- Complete backend API functional
- Dashboard displays all analyses correctly
- All integration tests passing
- Performance targets achieved

---


### Phase 3: Evaluation, User Study & Polish (September 2 – October 15, 2026)
**Duration**: 6 weeks | **Effort**: 60h

#### Deliverables:
1. **Evaluation Metrics** (early September)
   - Proximity accuracy (MAE < 0.15)
   - Clustering quality (Silhouette > 0.5, Purity > 0.80)
   - Hotspot detection (F1 > 0.75)
   - Emotion prediction accuracy (> 0.65)
   - Performance benchmarks

2. **Ablation Study** (mid September)
   - 7 experimental conditions (Full, Ablate-Geo, etc.)
   - Statistical analysis of results
   - Baseline comparisons (Geo-only, K-means, Random)
   - Results visualization and documentation

3. **User Study** (late September)
   - Protocol design
   - Recruit 10-15 mental health researchers
   - Conduct interviews/surveys
   - Analyze qualitative feedback
   - Document findings and recommendations

4. **Documentation & Demo** (early October)
   - Comprehensive demo script
   - Video demonstration
   - Case study analyses
   - Updated architecture documentation
   - API reference guide

**Milestones**:
- All evaluation metrics meet success criteria
- Ablation study confirms multi-dimensional approach
- User feedback validates usefulness
- Complete documentation ready for handoff

---


### Final Phase: Wrap-up & Submission (October 16 – November 11, 2026)
**Duration**: 4 weeks | **Effort**: 30h

#### Deliverables:
- GSoC final report
- Final presentation to mentors/community
- Code cleanup and refactoring
- Contributor guide for future developers
- Knowledge transfer documentation

**Milestone**: Project ready for production deployment

---

**Total GSoC Coding Effort (Phases 1–4): 350 hours**

---

## Post-GSoC Enhancements (Future Work)

### Short-term (3-6 months)

#### 1. Advanced Place Enrichment
**Description**: Integrate external APIs for automatic place type detection and tagging.

**Features**:
- Google Places API integration for:
  - Automatic place type inference from GPS
  - Business name and category extraction
  - Photo matching for location verification
- Nominatim (OpenStreetMap) as free alternative
- Automatic cultural tag extraction from place descriptions

**Benefits**:
- Reduces manual input burden
- Improves proximity accuracy with rich metadata
- Enables cross-user location matching

**Effort**: 40 hours

---

#### 2. Real-time Collaborative Features
**Description**: Enable cross-user emotion pattern analysis while preserving privacy.

**Features**:
- Anonymized aggregation of emotions at public locations
- "Others felt positive here too" insights
- Heatmap of community emotional landscape
- Privacy-preserving differential privacy techniques

**Benefits**:
- Social validation for recovery journeys
- Community-level mental health insights
- Research opportunities for population-level analysis

**Effort**: 60 hours

---

#### 3. Mobile App Integration
**Description**: Native mobile support for location-aware photo uploads.

**Features**:
- React Native mobile app
- Automatic GPS capture on photo upload
- Offline mode with sync
- Push notifications for emotional hotspot proximity
- Map-based photo browsing

**Benefits**:
- Improves data quality (native GPS access)
- Better user experience
- Real-time location-emotion tracking

**Effort**: 120 hours

---

### Medium-term (6-12 months)

#### 4. Predictive Emotion Modeling
**Description**: ML models to predict emotional responses at locations.

**Features**:
- Train models on historical location-emotion data
- Predict likely emotion at new/unvisited locations
- Personalized recommendations for emotionally beneficial places
- Transfer learning across users with similar patterns

**Benefits**:
- Proactive mental health support
- Personalized location recommendations
- Research insights into emotion-place associations

**Effort**: 80 hours

---

#### 5. Temporal Pattern Mining
**Description**: Advanced analysis of how location-emotion associations evolve.

**Features**:
- Change point detection in temporal trends
- Season/time-of-day emotion patterns
- Recovery milestone identification
- Longitudinal trajectory modeling

**Benefits**:
- Identify recovery inflection points
- Understand cyclical patterns
- Support intervention timing decisions

**Effort**: 60 hours

---

#### 6. Clinician Dashboard
**Description**: Specialized interface for therapists and researchers.

**Features**:
- Multi-patient aggregate view (with consent)
- Customizable reports for clinical sessions
- Export data in standard formats (CSV, FHIR)
- Annotation tools for clinical notes
- HIPAA compliance and data security

**Benefits**:
- Clinical research support
- Therapy integration
- Evidence-based interventions

**Effort**: 100 hours

---

### Long-term (1-2 years)

#### 7. Multi-modal Emotion Analysis
**Description**: Integrate additional data sources beyond photos/captions.

**Features**:
- Audio analysis (voice emotion detection)
- Video micro-expression analysis
- Wearable integration (heart rate, activity)
- Social media cross-posting analysis
- Calendar/schedule correlation

**Benefits**:
- Richer emotional context
- Triangulation of emotion signals
- Holistic recovery tracking

**Effort**: 200 hours

---

#### 8. Intervention Recommendation Engine
**Description**: AI-powered suggestions for mental health interventions.

**Features**:
- Pattern-based intervention recommendations
- "You felt better after visiting parks" insights
- Evidence-based coping strategies
- Connection to mental health resources
- Crisis detection and emergency protocols

**Benefits**:
- Actionable guidance for users
- Bridge between analysis and intervention
- Potential life-saving crisis support

**Effort**: 150 hours

---

#### 9. Research Platform & Data Sharing
**Description**: Enable academic research while protecting privacy.

**Features**:
- De-identified data export for researchers
- Federated learning across institutions
- Open dataset publication (with consent)
- Replication tools for published studies
- IRB-compliant data access workflows

**Benefits**:
- Accelerate mental health research
- Validate findings across populations
- Build evidence base for digital therapeutics

**Effort**: 120 hours

---

## Technology Evolution

### Current Stack
- **Backend**: Python Flask
- **Database**: MongoDB
- **ML**: Hugging Face Transformers, scikit-learn
- **Frontend**: Jinja2 templates, vanilla JavaScript
- **Visualization**: Leaflet.js (maps)

### Planned Upgrades

**Near-term**:
- **Frontend**: Migrate to React or Vue.js for richer interactivity
- **API**: GraphQL for flexible data querying
- **Caching**: Redis for proximity score caching
- **Task Queue**: Celery for async clustering jobs

**Long-term**:
- **Database**: TimescaleDB for better time-series performance
- **ML Ops**: MLflow for model versioning and deployment
- **Monitoring**: Prometheus + Grafana for production observability
- **Scaling**: Kubernetes for horizontal scaling

---

## Community & Ecosystem

### Open Source Growth
- **Contributors**: Attract additional GSoC students for future summers
- **Plugin System**: Allow third-party emotion analysis models
- **API Clients**: Official Python/JavaScript client libraries
- **Documentation**: Sphinx-generated API docs, video tutorials

### Integration Partnerships
- **Beehive**: Deeper integration with photo storytelling
- **Mental Health Apps**: Partnerships with existing platforms
- **Wearable Devices**: Official integrations with Fitbit, Apple Health
- **EHR Systems**: FHIR-compliant data export for clinical records

### Research Collaborations
- **University Partnerships**: Pilot studies with psychology departments
- **Funding**: Grants for large-scale clinical trials
- **Publications**: Research papers on location-emotion findings
- **Conferences**: Presentations at mental health informatics conferences

---

## Timeline Overview

```
2026
│
├─ Feb-Apr: Pre-GSoC Contributions (18 PRs)
├─ May-Aug: GSoC Implementation (350h)
├─ Sep-Dec: Production Deployment + Short-term Enhancements
│
2027
│
├─ Q1-Q2: Medium-term Features (Predictive Modeling, Temporal Mining)
├─ Q3-Q4: Long-term Features (Multi-modal Analysis)
│
2028+
│
└─ Research Platform, Intervention Engine, Ecosystem Growth
```

---

## Success Metrics (Long-term)

### Usage Metrics
- **Users**: 10,000+ active users by 2027
- **Photos Analyzed**: 1M+ photos with location data
- **Research Studies**: 10+ published papers using DREAMS data

### Impact Metrics
- **Recovery Outcomes**: Demonstrated improvement in recovery trajectories
- **Clinical Adoption**: 50+ clinicians actively using platform
- **Community Engagement**: Active open-source community (100+ stars)

### Technical Metrics
- **Performance**: 99.9% uptime, < 2s average response time
- **Scalability**: Support 100k+ concurrent users
- **Accuracy**: Emotion prediction accuracy > 0.75

---

**Version**: 1.0  
**Last Updated**: February 3, 2026  
**Author**: Krishan (GSoC 2026 Contributor)  
**Status**: Living document - updated quarterly
