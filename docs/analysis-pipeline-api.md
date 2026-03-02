# Analysis Pipeline API (Async Ingestion + Feature Extraction)

End-to-end documentation for the DREAMS analysis pipeline service that accepts real-world image uploads, runs feature extraction asynchronously, and exposes a JSON “analysis manifest” per memory for downstream research (Phase 1/2/3).

---

## Table of Contents

1. [What This Service Does](#what-this-service-does)
2. [Why This Runs Separately on Port 5001](#why-this-runs-separately-on-port-5001)
3. [Pipeline Overview](#pipeline-overview)
4. [Data Flow (Upload → Job → JSON)](#data-flow-upload--job--json)
5. [Storage Model (SQLite + ChromaDB)](#storage-model-sqlite--chromadb)
6. [API Reference](#api-reference)
7. [Analysis JSON Schema](#analysis-json-schema)
8. [Configuration](#configuration)
9. [File Map](#file-map)
10. [Running and Testing](#running-and-testing)
11. [Resetting State (Start Fresh)](#resetting-state-start-fresh)
12. [Known Limitations and Troubleshooting](#known-limitations-and-troubleshooting)
13. [Future Roadmap (Phase 1/2/3)](#future-roadmap-phase-123)

---

## What This Service Does

This service provides a production-oriented workflow for “memory” ingestion:

- Accept an image upload (optionally including caption, timestamp, GPS)
- Store the memory in SQLite
- Enqueue a background job immediately (fast HTTP response)
- Run feature extraction asynchronously in a worker thread
- Persist embeddings to ChromaDB and structured features to SQLite
- Expose a single JSON response per memory via `GET /api/analysis/<memory_id>`

The output JSON is designed to be the stable interface for later research modules:

- Phase 1: clustering / concept discovery (multimodal)
- Phase 2: sequential pattern analysis (temporal / transitions)
- Phase 3: prediction (future emotional/behavioral signals)

---

## Why This Runs Separately on Port 5001

DREAMS already contains an application server (commonly run on port 5000). This analysis pipeline runs separately (port 5001) for practical reasons:

1. Long-running ML steps (captioning, embeddings, multiple NLP models, geocoding) can take seconds to minutes on CPU, especially on first run when models download.
2. Keeping ML workloads separate avoids blocking user-facing request handling.
3. Failure isolation: memory pressure or model-download issues should not take down the main app.
4. Clean service boundary: this API’s contract is “upload → job status → analysis JSON”, which maps naturally to an async worker architecture.

Important note: running on a separate port is a deployment choice, not a hard requirement. The same blueprint could be mounted into the main app later; the async job model would remain.

---

## Pipeline Overview

```
Client (Postman / UI)
  │
  │  POST /api/ingest  (multipart image + metadata)
  ▼
Flask API (analysis_pipeline/api/app.py)
  │
  ├─ writes memory row to SQLite: analysis_pipeline/data/pipeline.db
  ├─ enqueues job row to SQLite ingest_queue
  └─ signals worker via threading.Event
  ▼
Background worker (analysis_pipeline/api/worker.py)
  │
  ├─ caption             (BLIP only if no user caption)
  ├─ image_embeddings    (CLIP → ChromaDB)
  ├─ caption_embeddings  (MiniLM → ChromaDB)
  ├─ emotions            (discrete + sentiment + CHIME; sequential model load)
  ├─ location            (reverse geocode with cache)
  └─ temporal            (cyclical time + relative day)
  ▼
Client polls status and fetches JSON
  │
  ├─ GET /api/status/<job_id>
  └─ GET /api/analysis/<memory_id>
```

---

## Data Flow (Upload → Job → JSON)

### Step 1 — Upload and Validation

**Endpoint:** `POST /api/ingest`

The API validates:

- `image` file exists and has a supported extension
- upload size is within the service limit (10 MB)
- `user_id` is present

The image is written to:

- `analysis_pipeline/data/processed/<memory_id>.<ext>`

Path traversal is prevented by validating output paths under allowed roots.

### Step 2 — Memory ID and Idempotency

This service generates a deterministic `memory_id` from:

- `user_id` and the uploaded filename

This means re-uploading the same filename for the same user can return `{"status": "already_exists"}`.

If you want a new `memory_id` for a repeated upload, change the filename (or extend the API later to incorporate a client-provided nonce).

### Step 3 — Duplicate Detection

A perceptual hash (average-hash) is computed from the image. If the new hash is within a small Hamming distance of an existing hash, the record is marked as a duplicate and is not processed.

### Step 4 — Enqueue and Background Processing

The API creates a queue row in the `ingest_queue` table (SQLite). It returns immediately:

- HTTP 202 with `{job_id, memory_id, status: "queued"}`

The worker thread sleeps when idle and wakes when signaled.

### Step 5 — Fetch the Analysis JSON

When the job finishes, request:

- `GET /api/analysis/<memory_id>`

This assembles fields from:

- `memories`
- `emotion_scores`
- `location_info`
- `temporal_features`
- `processing_state`
- ChromaDB collections for embeddings

---

## Storage Model (SQLite + ChromaDB)

### SQLite

SQLite is used for structured, queryable data and job state:

- DB path: `analysis_pipeline/data/pipeline.db`

Key tables:

- `memories`: one row per uploaded memory (image path, caption, GPS, timestamp, duplicates)
- `ingest_queue`: one row per processing job (queued / processing / done / error)
- `processing_state`: per-memory, per-step state (pending / done / error)
- `emotion_scores`: combined output of emotion models
- `location_info`: reverse geocode results
- `temporal_features`: cyclical and relative-day features

### ChromaDB

ChromaDB stores embedding vectors:

- Directory: `analysis_pipeline/data/chromadb/`

Collections used:

- `image_embeddings` (512D CLIP vectors)
- `caption_embeddings` (384D MiniLM vectors)

The analysis JSON includes either:

- references to collections and dimensions (default), or
- raw embedding vectors when requested with `?include_embeddings=true`

---

## API Reference

Base URL (local dev): `http://localhost:5001`

### POST /api/ingest

Upload a memory and enqueue background processing.

**Request**

- Content-Type: `multipart/form-data`
- Form fields:
  - `image` (file, required)
  - `user_id` (string, required)
  - `caption` (string, optional)
  - `latitude` (float, optional)
  - `longitude` (float, optional)
  - `category` (string, optional)
  - `timestamp` (string, optional; ISO-8601 recommended)

**Responses**

- `202 Accepted` (new job queued)
  - `{ "job_id": "...", "memory_id": "...", "status": "queued" }`

- `200 OK` (already exists)
  - `{ "job_id": null|"...", "memory_id": "...", "status": "already_exists" }`

- `200 OK` (duplicate)
  - `{ "memory_id": "...", "status": "duplicate", "duplicate_of": "..." }`

### GET /api/status/<job_id>

Poll job status.

**Response**

A row from `ingest_queue`, including:

- `status`: `queued` | `processing` | `done` | `error`
- `current_step`: name of the step currently running
- `error_message`: populated when `status=error`

### GET /api/analysis/<memory_id>

Fetch the full analysis JSON for one memory.

**Query parameters**

- `include_embeddings` (boolean; default `false`)

### GET /api/analysis

Paginated list of analysis records.

**Query parameters**

- `user_id` (optional)
- `page` (default 1)
- `per_page` (default 20; max 100)

---

## Analysis JSON Schema

The analysis response is designed to be a single, self-contained object that downstream analytics can consume without joining multiple tables.

Key fields:

- Identity: `memory_id`, `user_id`, `captured_at`, `category`
- Content: `caption`, `generated_caption`, `caption_source`
- Embeddings: either raw vectors or collection references
- Emotions:
  - discrete emotions (7-class probabilities)
  - sentiment (pos/neg/neutral)
  - CHIME category and confidence
- Location:
  - original GPS plus reverse-geocoded fields (address + place type)
- Temporal:
  - hour/dow/month/year + cyclical sin/cos + `recovery_day`
- Status:
  - `processing_status` and optional `processing_errors`

Example (abridged):

```json
{
  "memory_id": "7d52af1678e2557b",
  "user_id": "anishhhh",
  "captured_at": "2026-03-01T11:33:43.292329+00:00",
  "caption_source": "user",
  "emotions": {
    "dominant_emotion": "joy",
    "discrete": { "joy": 0.94, "neutral": 0.03 },
    "sentiment": { "label": "positive", "positive": 0.98 },
    "chime": { "category": "Hope", "confidence": 0.997 }
  },
  "location": {
    "latitude": 61.2181,
    "longitude": -149.9003,
    "place_type": "post_box",
    "address": { "city": "Anchorage", "state": "Alaska", "country": "United States" }
  },
  "temporal": {
    "hour": 11,
    "day_of_week": 6,
    "season": "spring",
    "recovery_day": 0.0,
    "cyclical": { "sin_hour": 0.26, "cos_hour": -0.97 }
  },
  "embeddings": {
    "image": { "collection": "image_embeddings", "dimensions": 512 },
    "caption": { "collection": "caption_embeddings", "dimensions": 384 }
  },
  "processing_status": "complete"
}
```

---

## Configuration

Core configuration lives in `analysis_pipeline/config.py`.

Notable settings:

- `SQLITE_DB_PATH`: `analysis_pipeline/data/pipeline.db`
- `CHROMA_DB_DIR`: `analysis_pipeline/data/chromadb`
- `BLIP_MODEL_NAME`: captioning model
  - The pipeline uses `Salesforce/blip-image-captioning-base` by default to reduce memory pressure.
- `IMAGE_EXTENSIONS`: allowed upload extensions

Operational behavior:

- Captioning is designed to be “lightweight by default”:
  - If a user caption exists, BLIP is not loaded.
  - The user caption is copied to `generated_caption` for downstream text steps.

---

## File Map

API layer:

- `analysis_pipeline/api/app.py`
  - Flask blueprint and `create_app()` factory
  - endpoints: ingest/status/analysis

- `analysis_pipeline/api/queue.py`
  - SQLite-backed queue table `ingest_queue`
  - `enqueue()`, `dequeue_batch()`, `recover_stale_jobs()`

- `analysis_pipeline/api/worker.py`
  - background worker thread
  - sleeps on `threading.Event` when queue is empty

- `analysis_pipeline/api/server.py`
  - entry point; configures logging; runs on port 5001

Pipeline steps:

- `analysis_pipeline/steps/caption.py`
- `analysis_pipeline/steps/image_embeddings.py`
- `analysis_pipeline/steps/caption_embeddings.py`
- `analysis_pipeline/steps/emotions.py`
- `analysis_pipeline/steps/location.py`
- `analysis_pipeline/steps/temporal.py`

Storage:

- `analysis_pipeline/db.py` (SQLite schema and helpers; ChromaDB helpers)
- `analysis_pipeline/utils.py` (timestamp parsing, hashing, path validation)

---

## Running and Testing

### Start the service

From the repository root:

```bash
python -m analysis_pipeline.api
```

The worker is started automatically.

### Test with Postman

1. Create `POST http://localhost:5001/api/ingest`
2. Body: `form-data`
3. Set:
   - `image` (File)
   - `user_id` (Text)
   - `caption` (Text, optional but recommended)
   - `latitude` / `longitude` (optional)
   - `timestamp` (optional)

Then poll:

- `GET http://localhost:5001/api/status/<job_id>`

Finally fetch:

- `GET http://localhost:5001/api/analysis/<memory_id>`

### Notes on first run

On the first run for a model, HuggingFace will download weights. This can take time and may retry on network timeouts.

Once cached, subsequent runs do not re-download.

---

## Resetting State (Start Fresh)

To wipe the pipeline state (SQLite + ChromaDB + uploaded/processed images), run:

```bash
python -m analysis_pipeline.erase_past_rec
```

This deletes the database and vector store directories and recreates the minimal folder structure.

---

## Known Limitations and Troubleshooting

### Memory / paging-file errors (Windows)

Symptom:

- `OSError: The paging file is too small for this operation to complete. (os error 1455)`

Root cause:

- Multiple large models loaded simultaneously.

Mitigations implemented:

- BLIP “base” is used instead of “large”.
- Models are unloaded after each step.
- Emotion models are loaded sequentially.

### Upload returns `already_exists`

Cause:

- `memory_id` is deterministic from `user_id` + filename.

Fix:

- Rename the file before uploading if you want it to be treated as a new memory.

### Geocoding is slow

Cause:

- Reverse geocoding is rate limited.

Fix:

- Results are cached in `analysis_pipeline/data/cache/geocode_cache.db`.
- Subsequent requests for the same rounded coordinates are fast.

### Timestamp parsing

The pipeline accepts many common timestamp formats. ISO-8601 is recommended.

---

## Future Roadmap (Phase 1/2/3)

1. **Phase 1** (multimodal clustering) will consume the per-memory JSON produced here.

Key design considerations:

- multimodal normalization and weighting when fusing image/text/location/time signals
- dimensionality reduction (e.g., PCA/UMAP) prior to clustering
- density-based clustering (e.g., HDBSCAN) for variable-density, noisy real-world data

The service’s primary goal is to provide stable, queryable feature extraction outputs so research can iterate independently of ingestion mechanics.


2. **Phase 2** : **Sequential Pattern Analysis (Order of Visits)**

Recovery is not a collection of isolated moments , it is a sequence. I want to take the discovered place concepts and the emotion labels, arrange them chronologically for each user, and analyze transition patterns. For example: "Does visiting an anxiety-inducing environment followed by a calming environment show a different emotional outcome than two anxiety-inducing visits in a row?" This would surface coping behaviors and risk patterns that are invisible when looking at individual photos independently.

3. **Phase 3** : **Emotional Prediction from Location Sequences**

Building on the first two phases, I want to train a lightweight sequence prediction model that learns the relationship between place-visit sequences and emotional outcomes. The goal is not clinical-grade diagnosis, but a proof of concept that shows: "Given a user's recent pattern of visiting certain types of places, can we estimate the likely emotional direction?"