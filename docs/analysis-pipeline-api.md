# Analysis Pipeline API (Async Ingestion + Feature Extraction)

Documentation for the DREAMS analysis pipeline service that accepts images uploads, runs feature extraction asynchronously, and exposes analysis JSON per memory for downstream tasks (Phase 1/2/3).

---

## Table of Contents

1. [What This Service Does](#what-this-service-does)
2. [Why It Runs in the Background](#why-it-runs-in-the-background)
3. [How It Works](#how-it-works)
4. [What Happens to Your Photos](#what-happens-to-your-photos)
5. [How Data is Stored](#how-data-is-stored)
6. [API Endpoints](#api-endpoints--how-to-use-it)
7. [The JSON You Get Back](#the-json-you-get-back)
8. [Settings and Configuration](#settings-and-configuration)
9. [File Organization](#file-organization)
10. [Starting and Testing](#starting-and-testing)
11. [Cleaning Everything Up](#cleaning-everything-up)
12. [Common Problems and Solutions](#common-problems-and-solutions)
13. [What's Being Built Next](#whats-being-built-next)

---

## What This Service Does

This service processes batches of memories (photos with captions and timestamps):

- Accept batch uploads (CSV metadata + ZIP of photos)
- Extract and validate all metadata from CSV
- Store memories in SQLite
- Enqueue background processing immediately (fast response)
- Extract features asynchronously (images, captions, emotions, temporal)
- Store vectors in ChromaDB and structured data in SQLite
- Expose complete analysis JSON for each memory

Each photo becomes a complete record with:
- Image and caption vectors (for similarity search)
- Emotion analysis (7 emotions + sentiment + recovery category)
- Temporal features (hour, day, season, cyclical patterns)
- Processing status and metadata

---

## Why It Runs in the Background

Processing batches takes time. Running embeddings, emotion analysis, and temporal feature extraction on many photos can take several seconds per photo.

The API:
- Accepts your batch and returns immediately with a batch ID
- Does all processing in the background
- Lets you check progress anytime
- Provides results when complete

This means you can upload large batches without waiting for each one to finish.

---

## How It Works

![alt text](image.jpg)

---

## Data Flow (Batch Processing → Jobs → JSON)

### Step 1 — Batch Upload and Validation

**Primary Endpoint:** `POST /api/ingest/batch`

```
CSV metadata arrives    → Rows validated (user_id, caption, timestamp ALL required)
ZIP photos arrive      → Extracted and checked (JPG, PNG, supported formats)
                       → All files checked for size limits (max 1GB total)
                       → Photos saved to: analysis_pipeline/data/processed/
```

The API validates:
- CSV has required columns: `filename`, `user_id`, `caption`, `timestamp`
- Each row's file exists in the ZIP
- All image files have supported extensions (.jpg, .png, .webp, .bmp, .tiff)
- Total uncompressed size is within limit (1 GB max for batch, max 1000 files per batch)

### Step 2 — Memory ID Creation and Batch Queuing

For each photo in the batch, the system creates a unique memory ID from its user ID and filename. 

- If you upload the exact same file twice, it's recognized as a duplicate
- Each valid file from the CSV gets queued as a separate processing job
- The API returns immediately with `batch_id` and `enqueued_count`

To treat a repeat upload as new, just rename the file in your CSV/ZIP before re-uploading.

### Step 3 — Duplicate Detection

For each photo in the batch, the system creates a "perceptual hash"—a fingerprint based on what the photo looks like:

- If this photo is too similar to an existing one, it's marked as a duplicate
- Duplicates are recorded but not processed  
- You'll see which photo it's a copy of

### Step 4 — Background Worker Processing

The queue manager picks up jobs from the batch and distributes them to the background worker:

- Worker runs processing steps for each photo independently
- Multiple photos can be processed in parallel (worker pool)
- HTTP 202 response means batch was accepted; processing happens in background

You check progress with:
- `GET /api/batch/<batch_id>/status` for overall batch progress
- `GET /api/status/<job_id>` for individual photo progress

### Step 5 — Worker Processing

The background worker runs these steps in order:
1. Image embeddings — Convert photo to a vector
2. Caption embeddings — Convert description to a vector  
3. Emotions — Analyze feelings
4. Temporal — Extract time patterns

Each step saves results to the database.

### Step 6 — Fetch Complete Analysis Results

When a photo's processing is done, request its complete analysis:

- `GET /api/analysis/<memory_id>`

This assembles all results (emotions, embeddings, temporal features) into one JSON object ready to use.

You can also fetch a paginated list of all analyses:

- `GET /api/analysis` with optional filters (`user_id`, `page`, `per_page`)

---

## Storage Model (SQLite + ChromaDB)

### The Database (SQLite)

All structured information lives in: `analysis_pipeline/data/pipeline.db`

**Main tables:**
- **memories** — The core record for each photo (filename, caption, when it was taken, duplicate status)
- **emotion_scores** — All the feeling analysis (7 discrete emotions, mood, sentiment, CHIME recovery category)
- **temporal_features** — Time-based information (hour, day of week, season, special patterns)
- **processing_state** — Tracks what's been done and what's not (completed steps, errors, timestamps)

### Vector Storage (ChromaDB)

Numerical representations of photos and text live in: `analysis_pipeline/data/chromadb/`

Two collections:
- **image_embeddings** — Photos converted to 512-number vectors (CLIP model)
- **caption_embeddings** — Written descriptions converted to 384-number vectors (MiniLM model)

These vectors let the system find similar memories without reading text or looking at images directly.

---

## API Endpoints — How to Use It

### Base URL

When running locally: `http://localhost:5001`

### Upload Many Photos (Recommended)

**Endpoint:** `POST /api/ingest/batch`

Upload a CSV file and a ZIP of photos together for bulk processing.

- `csv` (required) — CSV file with columns: `filename`, `user_id`, `caption`, `category` (optional), `timestamp`
- `images` (required) — ZIP archive containing the photo files

Response:
```json
{
  "status": "accepted",
  "batch_id": "batch_abc123xyz",
  "enqueued_count": 50
}
```

### Upload a Single Photo (Legacy)

**Endpoint:** `POST /api/ingest`

For single photo uploads (now primarily used for testing):

- `image` (required) — The photo file
- `user_id` (required) — Your identifier (e.g., "anishhhh")
- `caption` (required) — Your description of the photo
- `category` (optional) — A label like "park", "hospital", "home"
- `timestamp` (required) — When the photo was taken (ISO-8601 format)

### Check if a Job is Done

**Endpoint:** `GET /api/status/<job_id>`

Response:
```json
{
  "job_id": "a1b2c3d4e5f6g7h8",
  "memory_id": "uniqueid123",
  "status": "processing",
  "current_step": "emotions",
  "error_message": null
}
```

Possible statuses:
- `queued` — Waiting to start
- `processing` — Currently running
- `done` — All finished
- `error` — Something went wrong

### Check a Batch Upload Status

**Endpoint:** `GET /api/batch/<batch_id>/status`

Shows counts for queued, processing, done, and error statuses.

### Get the Complete Analysis for One Photo

**Endpoint:** `GET /api/analysis/<memory_id>`

Optional query:
- `include_embeddings=true` or `false` (default is `true`)
  - `true` = includes the actual number arrays
  - `false` = just tells you where to find them

### Get a List of All Photos

**Endpoint:** `GET /api/analysis`

Optional query parameters:
- `user_id=anishhhh` — Filter to just your photos
- `page=2` — Which page of results
- `per_page=20` — How many per page (max 100)

Response:
```json
{
  "total": 250,
  "page": 1,
  "per_page": 20,
  "records": [
    { ... full analysis for first photo ... },
    { ... full analysis for second photo ... }
  ]
}
```

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
  - `caption` (string, required)
  - `category` (string, optional)
  - `timestamp` (string, required; ISO-8601 recommended)

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

- `include_embeddings` (boolean; default `true`)

### GET /api/analysis

Paginated list of analysis records.

**Query parameters**

- `user_id` (optional)
- `page` (default 1)
- `per_page` (default 20; max 100)

---

## The JSON You Get Back

When you fetch `GET /api/analysis/<memory_id>`, you get one complete object with everything about that memory:

```json
{
  "memory_id": "uniqueid123",
  "user_id": "anishhhh",
  "image_path": "/path/to/photo.jpg",
  "category": "park",
  "caption": "I wrote this myself",
  "caption_source": "user",
  "captured_at": "2026-03-01T14:30:00+00:00",
  "is_duplicate": false,
  
  "emotions": {
    "discrete": {
      "anger": 0.02,
      "disgust": 0.01,
      "fear": 0.03,
      "joy": 0.80,
      "neutral": 0.05,
      "sadness": 0.05,
      "surprise": 0.04
    },
    "dominant_emotion": "joy",
    "valence": 0.72,
    "arousal": 0.45,
    "sentiment": {
      "label": "positive",
      "positive": 0.92,
      "negative": 0.03,
      "neutral": 0.05
    },
    "chime": {
      "category": "Hope",
      "confidence": 0.95
    }
  },
  
  "temporal": {
    "hour": 14,
    "day_of_week": 2,
    "month": 3,
    "year": 2026,
    "season": "spring",
    "time_of_day": "afternoon",
    "recovery_day": 125.5,
    "cyclical": {
      "sin_hour": 0.34,
      "cos_hour": -0.94,
      "sin_dow": 0.95,
      "cos_dow": 0.31,
      "sin_month": 0.50,
      "cos_month": 0.87
    }
  },
  
  "embeddings": {
    "image": {
      "collection": "image_embeddings",
      "dimensions": 512
    },
    "caption": {
      "collection": "caption_embeddings",
      "dimensions": 384
    }
  },
  
  "processing_status": "complete",
  "steps_completed": ["image_embeddings", "caption_embeddings", "emotions", "temporal"]
}
```

### Understanding the Fields

**Emotions section:**
- **discrete** — Scores (0 to 1) for each of 7 basic feelings. Higher = more of that feeling
- **dominant_emotion** — Whichever emotion scored highest
- **valence** — Scale from -1 (unhappy) to +1 (happy). How pleasant is this moment?
- **arousal** — Scale from 0 (calm) to 1 (energetic). How active is this emotion?
- **sentiment** — Overall: positive, negative, or neutral? With confidence scores for each
- **chime** — Recovery category from DREAMS's model (Hope, Anxiety, Sadness, Acceptance, etc.)

**Temporal section:**
- **hour, day_of_week, month, year** — Straightforward time information
- **season** — winter, spring, summer, or fall  
- **time_of_day** — morning, afternoon, evening, or night
- **recovery_day** — How many days since your first memory? (Useful for tracking progress)
- **cyclical** — Mathematical transformations that make time patterns circular
  - Hour 23 and hour 1 are treated as close together
  - Same logic for days, weeks, months
  - These numbers are used by algorithms to understand repeating patterns

**Embeddings section:**
- These vectors are mathematical representations of your photo and its description
- Used to find similar memories later
- Image vectors are 512 numbers, captions are 384 numbers
- Can be expensive to transmit, so by default you just get metadata about where they're stored

---

## Settings and Configuration

Core settings live in: `analysis_pipeline/config.py`

**Models used:**
- **Image vectors:** OpenAI CLIP ViT-B/32 (about 340 MB)
- **Caption vectors:** MiniLM-L6-v2 (about 80 MB)
- **Emotion analysis:** Multiple models, loaded one at a time

**Key limits:**
- Single photo: max 10 MB
- Batch: max 1 GB total

---

## File Organization

```
analysis_pipeline/
├── api/
│   ├── app.py              ← Flask app with endpoints
│   ├── worker.py           ← Background worker
│   ├── queue.py            ← Queue management
│   └── server.py           ← Starts service on port 5001
│
├── steps/
│   ├── image_embeddings.py ← CLIP image vectors
│   ├── caption_embeddings.py ← MiniLM text vectors
│   ├── emotions.py         ← Emotion analysis
│   ├── temporal.py         ← Time patterns
│   └── ingest.py           ← CSV/folder import
│
└── data/
    ├── processed/          ← Uploaded photos
    ├── chromadb/           ← Vector storage
    └── pipeline.db         ← SQLite database
```

---

## Starting and Testing

### Start the service

**From the repository root:**

```bash
python -m analysis_pipeline.api
```

You should see:
```
INFO: Server running on http://0.0.0.0:5001
```

The worker and API start automatically.

### Quick test with a sample photo

**Using any HTTP client (Postman, curl, etc.):**

```
POST http://localhost:5001/api/ingest

Form fields:
  image: [select any JPG or PNG file]
  user_id: test_user
  caption: A brief description
  timestamp: 2026-03-01T14:30:00Z
```

You'll get back a `job_id`. Check progress:

```
GET http://localhost:5001/api/status/<job_id>
```

Once done, fetch results:

```
GET http://localhost:5001/api/analysis/<memory_id>
```

### First time setup notes

On the first run, some models download from the internet (50 MB to 1 GB):
- This might take 1-3 minutes
- Subsequent runs use the cached version
- If network is slow, the pipeline retries automatically

---

## Cleaning Everything Up

To start from scratch:

```bash
python -m analysis_pipeline.erase_past_rec
```

Deletes database, vectors, and photos. Folder structure is recreated.

---

## Common Problems and Solutions


### Upload returns `already_exists`

Cause:

- `memory_id` is deterministic from `user_id` + filename.

Fix:

- Rename the file before uploading if you want it to be treated as a new memory.

### Timestamp parsing

The pipeline accepts many common timestamp formats. ISO-8601 is recommended.

---

## What's Being Built Next

### Phase 2: Similarity Modeling & Semantic Clustering
While Phase 1 focused on extracting raw features, Phase 2 implements the "Sense of Place" logic.
- **Semantic Similarity Mapping**: Utilizing ChromaDB distance metrics to group memories into visually and contextually similar "Place Groups".

### Phase 3: Temporal Analysis of Emotional Trajectories
This phase moves from static snapshots to dynamic recovery patterns.
- **Sequence Modeling**: Analyzing the transitions between emotional states. 
- **Path Visualization**: Mapping the user's emotional movement over days and weeks to visualize progress in their recovery journey.
