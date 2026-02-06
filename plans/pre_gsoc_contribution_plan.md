# Pre-GSoC Contribution Plan for DREAMS: Multi-Dimensional Location Proximity and Emotion Analysis

## Overview

This pre-GSoC contribution plan outlines 18 pull requests (PRs) spread over 7 weeks, designed to strengthen the DREAMS project's foundation in multi-dimensional location proximity and emotion analysis. The plan focuses on enhancing the Flask backend with MongoDB integration, building upon existing modules like `exif_extractor` and `location_proximity`, while ensuring alignment with the GSoC 2026 proposal.

**Recent Contributions by Other Developers**:
- **PR #77** (kunal-595): EXIF GPS extraction implemented - we integrate with this
- **PR #70** (AnvayKharb): Time-aware emotion proximity - we complement this with spatial proximity
- **PR #79** (anish1206): CHIME mental health framework - our emotion-location work aligns with CHIME dimensions

**Total Duration**: 7 weeks  
**Total PRs**: 18  
**Focus Areas**: Architecture (4 PRs), Research (3 PRs), Interfaces (4 PRs), Testing (4 PRs), Proposal Alignment (3 PRs)  
**Key Technologies**: Python Flask, MongoDB, scikit-learn, Hugging Face Transformers  

---

## Weekly Breakdown

### Week 1: Research Foundation and Architecture Setup
**Focus**: Establish research base and architectural foundations.

#### PR 1: Research Literature Review Update
**Description**: Expand the research foundation in `location_proximity/RESEARCH.md` with additional literature on affective geography and semantic similarity. Add references to recent papers on emotion-location associations in mental health recovery.
**Dependencies**: None
**Deliverables**:
- Updated `location_proximity/RESEARCH.md` with 5+ new citations
- Summary of key findings in `docs/research_summary.md`

#### PR 2: Architecture Documentation Refinement
**Description**: Refine `ARCHITECTURE.md` to include detailed MongoDB schema designs for location-emotion data storage. Update Mermaid diagrams to reflect MongoDB integration points.
**Dependencies**: PR 1
**Deliverables**:
- Enhanced `ARCHITECTURE.md` with MongoDB-specific sections
- New schema diagrams for location and emotion collections

#### PR 3: Database Schema Implementation
**Description**: Implement MongoDB schemas in `dreamsApp/app/models.py` for storing location proximity data and emotion-location mappings. Ensure compatibility with existing post schema.
**Dependencies**: PR 2
**Deliverables**:
- Updated `dreamsApp/app/models.py` with new MongoDB collections
- Migration scripts for schema updates

### Week 2: Core Location Proximity Enhancements
**Focus**: Strengthen core proximity calculation modules.

#### PR 4: Enhanced EXIF Location Extraction
**Status**: **Completed by PR #77** (kunal-595) - EXIF extractor already implemented in `dreamsApp/exif_extractor.py`
**Description**: ~~Improve `dreamsApp/exif_extractor.py`~~ Integration with existing EXIF extractor to ensure compatibility with multi-dimensional proximity module.
**Dependencies**: PR 3
**Deliverables**:
- ~~Enhanced `dreamsApp/exif_extractor.py` with better error handling~~ Already exists
- Integration tests with existing `EXIFExtractor` class

#### PR 5: Multi-Dimensional Proximity Calculator Refinement
**Description**: Refine `location_proximity/proximity_calculator.py` to optimize weighted proximity calculations and add configurable dimension weights.
**Dependencies**: PR 4
**Deliverables**:
- Updated `location_proximity/proximity_calculator.py`
- Performance benchmarks for proximity calculations

#### PR 6: Emotion-Location Mapper Implementation
**Description**: Complete implementation of `location_proximity/emotion_location_mapper.py` with methods for temporal emotion trends and hotspot identification.
**Dependencies**: PR 5
**Deliverables**:
- Functional `location_proximity/emotion_location_mapper.py`
- Integration with sentiment analysis from `dreamsApp/app/utils/sentiment.py`

### Week 3: Semantic Clustering and Interface Development
**Focus**: Implement clustering algorithms and initial API interfaces.

#### PR 7: Semantic Clustering Enhancements
**Description**: Enhance `location_proximity/semantic_clustering.py` with improved DBSCAN parameters and add visualization support for clusters.
**Dependencies**: PR 6
**Deliverables**:
- Updated `location_proximity/semantic_clustering.py`
- Clustering quality metrics implementation

#### PR 8: REST API Endpoints for Location Analysis
**Description**: Add new API endpoints in `dreamsApp/app/ingestion/routes.py` for proximity calculations and location-emotion queries.
**Dependencies**: PR 7
**Deliverables**:
- New routes in `dreamsApp/app/ingestion/routes.py`
- API documentation updates

#### PR 9: Dashboard UI Components for Location Proximity
**Description**: Create new dashboard templates in `dreamsApp/app/templates/dashboard/` for visualizing location proximity patterns.
**Dependencies**: PR 8
**Deliverables**:
- New HTML templates for location analysis
- Basic JavaScript for map visualizations

### Week 4: Emotion Analysis Integration and Testing
**Focus**: Integrate emotion analysis and begin comprehensive testing.

#### PR 10: Sentiment Analysis Integration with Locations
**Description**: Integrate emotion analysis from `dreamsApp/app/utils/sentiment.py` with location data in the ingestion pipeline.
**Dependencies**: PR 9
**Deliverables**:
- Updated ingestion routes with emotion-location mapping
- Data flow integration in `dreamsApp/app/ingestion/routes.py`

#### PR 11: Unit Tests for Location Proximity Modules
**Description**: Create comprehensive unit tests for all location proximity components in `tests/test_location_proximity.py`.
**Dependencies**: PR 10
**Deliverables**:
- Complete test suite in `tests/test_location_proximity.py`
- Test data fixtures for locations and emotions

#### PR 12: Integration Tests for Location-Emotion Pipeline
**Description**: Develop integration tests covering the full pipeline from image upload to emotion-location analysis.
**Dependencies**: PR 11
**Deliverables**:
- New integration test file `tests/test_location_emotion_integration.py`
- End-to-end test scenarios

### Week 5: Performance Optimization and Documentation
**Focus**: Optimize performance and enhance documentation.

#### PR 13: Performance Optimization for Proximity Calculations
**Description**: Implement caching and batch processing optimizations for proximity calculations in the Flask app.
**Dependencies**: PR 12
**Deliverables**:
- Caching layer in `dreamsApp/app/utils/`
- Performance improvements documentation

#### PR 14: Comprehensive Documentation Updates
**Description**: Update all README files and create user guides for location proximity features.
**Dependencies**: PR 13
**Deliverables**:
- Updated `README.md`, `location_proximity/README.md`
- User guide in `docs/location_proximity_guide.md`

#### PR 15: Demo Script and Example Improvements
**Description**: Enhance the demo script in `location_proximity/demo.py` with more comprehensive examples and better output formatting.
**Dependencies**: PR 14
**Deliverables**:
- Improved `location_proximity/demo.py`
- Sample data for demonstrations

### Week 6: Advanced Features and Validation
**Focus**: Implement advanced features and validation metrics.

#### PR 16: Validation Metrics and Statistical Analysis
**Description**: Implement validation metrics from the research foundation, including clustering quality and emotion prediction accuracy.
**Dependencies**: PR 15
**Deliverables**:
- New module `location_proximity/validation_metrics.py`
- Statistical analysis functions

#### PR 17: Cross-User Location-Emotion Analysis
**Description**: Add features for analyzing location-emotion patterns across multiple users while maintaining privacy.
**Dependencies**: PR 16
**Deliverables**:
- Cross-user analysis functions in `location_proximity/emotion_location_mapper.py`
- Privacy-preserving aggregation methods

### Week 7: Final Integration and Proposal Alignment
**Focus**: Finalize integrations and ensure proposal compliance.

#### PR 18: Final Proposal Alignment and Integration Testing
**Description**: Conduct final review to ensure all contributions align with GSoC 2026 proposal requirements. Perform comprehensive integration testing.
**Dependencies**: PR 17
**Deliverables**:
- Proposal alignment checklist
- Final integration test results
- Updated project roadmap

---

## Dependencies and Prerequisites

- **Technical Prerequisites**: Python 3.8+, Flask, MongoDB, scikit-learn, Pillow
- **Project Knowledge**: Familiarity with DREAMS architecture and existing modules
- **Testing Environment**: Access to test MongoDB instance and sample image data

## Success Metrics

- All 18 PRs merged successfully
- 90%+ test coverage for new location proximity code
- Performance benchmarks meeting requirements
- Documentation completeness for all new features
- Alignment with GSoC proposal objectives

## Risk Mitigation

- Weekly code reviews to catch integration issues early
- Incremental testing to ensure stability
- Regular alignment checks with project mentors
- Backup plans for complex PRs with multiple dependencies

---

**Plan Created**: December 2025  
**Total Estimated Effort**: 18 PRs across 7 weeks  
**Primary Contributor**: Krishan (Pre-GSoC Contributor)