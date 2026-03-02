# DREAMS Analysis Pipeline

A production-ready, fault-tolerant feature extraction pipeline for the DREAMS project.

## Quick Start

```bash
# install dependencies
pip install -r analysis_pipeline/requirements.txt

# run full pipeline on a CSV dataset
python -m analysis_pipeline path/to/dataset.csv

# run with CSV export of the master manifest
python -m analysis_pipeline path/to/dataset.csv --export output/manifest.csv

# run only specific steps
python -m analysis_pipeline path/to/dataset.csv --only ingest emotions temporal

# skip heavy ML steps during development
python -m analysis_pipeline path/to/dataset.csv --skip image_embeddings caption
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| `ingest` | Import CSV into SQLite with deduplication and snapshotting |
| `caption` | Generate BLIP captions for images without user captions |
| `image_embeddings` | CLIP ViT-B/32 image vectors → ChromaDB |
| `caption_embeddings` | MiniLM sentence vectors for captions → ChromaDB |
| `emotions` | 7 discrete emotions + valence/arousal + sentiment + CHIME |
| `location` | Reverse-geocode GPS → place labels (with response caching) |
| `temporal` | Cyclical time encoding + relative recovery day |
| `manifest` | Quality report + optional CSV/Parquet export |

## Key Improvements

### Over prior pipeline

| Feature | Prior Pipeline | This Pipeline |
|---------|---------------|---------------|
| **Error recovery** | Step-level (crash = redo everything) | Per-record (crash = resume from last record) |
| **Duplicate detection** | None | Perceptual hashing detects near-identical photos |
| **Geocode caching** | None (re-hits API every run) | SQLite cache (each coordinate queried only once) |
| **Emotion richness** | 7 discrete only | 7 discrete + valence/arousal + sentiment + CHIME |
| **Data import** | Hardcoded to one dataset | Flexible CSV format with EXIF fallback |
| **Idempotency** | Re-inserts duplicates | Deterministic IDs, safe re-run |
| **Logging** | Console only | Console + timestamped log files |
| **Quality validation** | Row count check only | Completeness %, NULL detection, error summary |

### Architecture

```
CSV / Images
    │
    ▼
┌─────────────────────────────────────────────┐
│  INGEST                                      │
│  • Deterministic memory_id                   │
│  • Perceptual hash → duplicate detection     │
│  • EXIF GPS fallback                         │
│  • Versioned data snapshots                  │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    ┌─────────┐ ┌────────┐ ┌──────────┐
    │ CAPTION │ │ IMAGE  │ │ LOCATION │
    │ (BLIP)  │ │ EMBED  │ │ (geocode │
    │         │ │ (CLIP) │ │  +cache) │
    └────┬────┘ └───┬────┘ └────┬─────┘
         │         │           │
         ▼         │           │
    ┌─────────┐    │           │
    │ CAPTION │    │           │
    │ EMBED   │    │           │
    │ (MiniLM)│    │           │
    └────┬────┘    │           │
         │         │           │
         ▼         ▼           ▼
    ┌──────────────────────────────────────┐
    │  EMOTIONS                             │
    │  • 7 discrete (distilroberta)         │
    │  • Valence / arousal                  │
    │  • Sentiment (pos/neg/neu)            │
    │  • CHIME recovery category            │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  TEMPORAL                             │
    │  • Cyclical sin/cos encoding          │
    │  • Time-of-day / season buckets       │
    │  • Relative recovery day              │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  MANIFEST                             │
    │  • Quality report (completeness %)    │
    │  • Error summary per step             │
    │  • Optional CSV export                │
    └──────────────────────────────────────┘
```

## Data Storage

| Store | Purpose |
|-------|---------|
| **SQLite** (`pipeline.db`) | Structured data: memories, emotions, temporal features, location info, processing state |
| **ChromaDB** (`chromadb/`) | Vector embeddings: image_embeddings, caption_embeddings |
| **SQLite** (`geocode_cache.db`) | Cached Nominatim responses (never re-queries same coordinates) |

## Expected CSV Format

```csv
id,user_id,image_filename,category,latitude,longitude,date,caption
1,user_123,photo.jpg,Park,61.2058,-149.9141,2026-03-07 11:00:00,"Walked around the park today"
```

- `id`: row identifier (optional, not used as primary key)
- `user_id`: user identifier (required)
- `image_filename`: image file name, searched in same dir as CSV and `images/` subfolder (required)
- `category`: place category label (optional)
- `latitude` / `longitude`: GPS coordinates (optional — EXIF fallback)
- `date`: timestamp in ISO format (optional)
- `caption`: user-written description (optional — BLIP generates if missing)
