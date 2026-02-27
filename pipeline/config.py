"""
Configuration constants for DREAMS Research pipeline.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Phase 1: Raw data paths
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "images"
RAW_METADATA_PATH = RAW_DIR / "metadata.json"

# Processed data directory
PROCESSED_DIR = DATA_DIR / "processed"

# Pipeline logs
LOG_DIR = PROCESSED_DIR / "logs"

# SQLite database
DREAMS_DB_PATH = PROCESSED_DIR / "dreams.db"

# ChromaDB
CHROMA_DB_DIR = PROCESSED_DIR / "chroma_db"
IMAGE_COLLECTION_NAME = "image_embeddings"
CAPTION_COLLECTION_NAME = "caption_embeddings"
LOCATION_COLLECTION_NAME = "location_descriptions"

# Snapshots
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

# Model configurations (Phase 2)
CLIP_MODEL = "ViT-B/32"
SENTENCE_BERT_MODEL = "all-MiniLM-L6-v2"
EMOTION_MODEL = "Mavdol/NPC-Valence-Arousal-Prediction"
DISCRETE_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# D1 Database Schema
D1_SCHEMA = {
    "table": "memories",
    "columns": [
        "id",
        "user_id",
        "caption",
        "timestamp",
        "lat",
        "lon",
        "image_url",
        "processed",
        "processing_version",
        "created_at"
    ]
}

# Discrete emotion labels
DISCRETE_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Pipeline steps in execution order (name, description)
PIPELINE_STEPS = [
    ("import",              "Ingest CSV or D1 data into SQLite"),
    ("emotions",            "Extract valence/arousal + discrete emotion scores"),
    ("temporal",            "Compute cyclical time features"),
    ("location_embeddings", "Geocode + CLIP multi-modal location embeddings"),
    ("caption_embeddings",  "Sentence-BERT caption embeddings"),
    ("image_embeddings",    "CLIP image embeddings"),
    ("verify",              "Verify table/vector alignment"),
    ("manifest",            "Report master manifest statistics"),
]

