# DREAMS Research

Computational pipeline for validating **Stable Emotional Fingerprints** in human memory. We test whether specific locations induce statistically consistent emotional states over time.

## Hypothesis

Physical locations possess a stable emotional fingerprint. When a user visits the same place repeatedly, their emotional state converges to a consistent pattern — measurable via mean emotional vector (μ), covariance (Σ), and entropy (H).

## Pipeline

Single-command orchestrator that transforms raw memory logs into research-ready vectors.

```bash
# Full pipeline
python pipeline/run_pipeline.py clinical_depression_study/dataset.csv

# Or use Make
make pipeline
```

### Steps

| # | Step | Model / Tool | Output |
|---|------|-------------|--------|
| 1 | **Import** | — | CSV → SQLite `memories` |
| 2 | **Emotions** | `Mavdol/NPC-Valence-Arousal-Prediction` + `j-hartmann/emotion-english-distilroberta-base` | SQLite `emotion_scores` |
| 3 | **Temporal** | sin/cos hour encoding | SQLite `temporal_features` |
| 4 | **Location Embeddings** | Nominatim + CLIP `ViT-B/32` (text+image fusion) | ChromaDB + SQLite `location_descriptions` |
| 5 | **Caption Embeddings** | Sentence-BERT `all-MiniLM-L6-v2` | ChromaDB `caption_embeddings` |
| 6 | **Image Embeddings** | CLIP `ViT-B/32` | ChromaDB `image_embeddings` |
| 7 | **Verify** | — | Alignment check |
| 8 | **Manifest** | — | `master_manifest` view report |

### CLI Options

```bash
python pipeline/run_pipeline.py dataset.csv --only emotions,temporal   # run specific steps
python pipeline/run_pipeline.py dataset.csv --skip location_embeddings # skip slow steps
python pipeline/run_pipeline.py dataset.csv --resume                   # resume after crash
python pipeline/run_pipeline.py dataset.csv --export                   # export parquet
python pipeline/run_pipeline.py --source d1                            # pull from Cloudflare D1
```

## Data Schema

### SQLite (`data/processed/dreams.db`)

| Table / View | Description |
|:---|:---|
| `memories` | Raw metadata (user, caption, timestamp, GPS, image path) |
| `emotion_scores` | Valence, arousal, + 7 discrete emotion probabilities |
| `temporal_features` | Circadian sin/cos + relative day |
| `location_descriptions` | Geocoded location text + display name |
| `master_manifest` | **VIEW** — unified join of all tables |

### ChromaDB (`data/processed/chroma_db/`)

| Collection | Dim | Description |
|:---|:---|:---|
| `image_embeddings` | 512 | CLIP visual embeddings |
| `caption_embeddings` | 384 | S-BERT narrative embeddings |
| `location_descriptions` | 512 | CLIP text+image fused location embeddings |

## Setup

```bash
git clone https://github.com/ayusrjn/dreams-research.git
cd dreams-research
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU (recommended).
