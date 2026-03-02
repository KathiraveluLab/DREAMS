"""
DREAMS Analysis Pipeline
========================

A production-ready, fault-tolerant data processing pipeline for the DREAMS
project. Extracts visual, textual, emotional, and temporal features from
geotagged photos of recovery journeys.

Improvements over prior implementations:
- Per-record checkpointing and resume (never reprocesses completed records)
- Image deduplication via perceptual hashing
- Geocoder response caching (avoids redundant API calls)
- Richer emotion analysis (7 discrete + valence/arousal + CHIME recovery)
- Batch processing with progress tracking
- Flexible ingestion (CSV, image folder, JSON)
- Embedding quality validation (not just row-count checks)
- Per-record error isolation (one bad record doesn't kill the pipeline)
"""

__version__ = "0.1.0"
