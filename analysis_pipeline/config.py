"""
Centralised configuration for the analysis pipeline.

All paths, model identifiers, and tuneable parameters live here so that
every other module imports from a single source of truth.
"""

from pathlib import Path
import os

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
LOG_DIR = ROOT_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

# ensure directories exist on import
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, SNAPSHOT_DIR, LOG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
SQLITE_DB_PATH = DATA_DIR / "pipeline.db"
CHROMA_DB_DIR = DATA_DIR / "chromadb"

# ── AI / ML model identifiers ────────────────────────────────────────────────
# Image embeddings (HuggingFace CLIPModel — no separate 'clip' package needed)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Caption generation (base ≈ 990 MB vs large ≈ 1.8 GB — prevents OOM on <8 GB RAM)
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Text / sentence embeddings
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"

# Emotion analysis  (discrete 7-class)
DISCRETE_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Emotion analysis  (valence-arousal regression)
VA_EMOTION_MODEL = "Mavdol/NPC-Valence-Arousal-Prediction"

# CHIME recovery classifier (if available)
CHIME_MODEL_NAME = "ashh007/dreams-chime-bert"

# Sentiment (pos/neg/neutral)
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ── Discrete emotion labels ──────────────────────────────────────────────────
DISCRETE_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

# ── Processing parameters ────────────────────────────────────────────────────
BATCH_SIZE = 16                   # records per processing batch
MAX_RETRIES = 3                   # per-record retry limit
IMAGE_HASH_SIZE = 16              # perceptual hash grid size (16 → 256 bits)
SIMILARITY_THRESHOLD = 0.90       # cosine threshold for duplicate detection

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# ── Pipeline step registry ────────────────────────────────────────────────────
PIPELINE_STEPS = [
    {
        "name": "ingest",
        "description": "Import data from CSV / folder / JSON into SQLite",
    },
    {
        "name": "caption",
        "description": "Generate captions for images using BLIP (if missing)",
    },
    {
        "name": "image_embeddings",
        "description": "Compute CLIP image embeddings → ChromaDB",
    },
    {
        "name": "caption_embeddings",
        "description": "Compute MiniLM sentence embeddings for captions → ChromaDB",
    },
    {
        "name": "emotions",
        "description": "Extract 7 discrete emotions + valence/arousal + sentiment + CHIME",
    },
    {
        "name": "temporal",
        "description": "Extract cyclical temporal features from timestamps",
    },
    {
        "name": "manifest",
        "description": "Build master manifest & run quality report",
    },
]
