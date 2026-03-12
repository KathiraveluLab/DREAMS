"""
Shared utility functions used across pipeline steps.

Path validation, image loading, perceptual hashing, text cleaning,
timestamp parsing.
"""

import hashlib
import logging
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Any

from PIL import Image

from .config import IMAGE_EXTENSIONS, IMAGE_HASH_SIZE

logger = logging.getLogger(__name__)


# ── Path validation (security) ────────────────────────────────────────────────

def validate_safe_path(
    path: str | Path,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """
    Resolve a path and verify it does not escape allowed directories.

    Prevents directory-traversal attacks (e.g. ``../../etc/passwd``).
    When *allowed_roots* is ``None`` the resolved path is returned as-is
    (still resolves symlinks, so callers always get an absolute path).

    Raises:
        ValueError: if the resolved path is not under any allowed root.
    """
    resolved = Path(path).resolve()
    if allowed_roots:
        for root in allowed_roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except ValueError:
                continue
        raise ValueError(
            f"Path {str(path)!r} resolves to {resolved}, "
            f"which is outside allowed directories"
        )
    return resolved


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> Image.Image:
    """Load an image from a local path and convert to RGB."""
    img = Image.open(str(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


# ── Perceptual hashing (for duplicate detection) ──────────────────────────────

def perceptual_hash(image: Image.Image, hash_size: int = IMAGE_HASH_SIZE) -> str:
    """Compute an average-hash fingerprint for an image.

    Two visually identical (or near-identical) photos will produce the
    same hash string, enabling cheap duplicate detection without
    computing full embeddings.
    """
    # resize to hash_size x hash_size and convert to greyscale
    img = image.resize((hash_size, hash_size), Image.LANCZOS).convert("L")
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    # each bit: 1 if pixel >= average, else 0
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    # convert bitstring to hex for compact storage
    return hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Number of differing bits between two hex-encoded perceptual hashes."""
    if len(hash_a) != len(hash_b):
        raise ValueError("Hashes must be the same length")
    bin_a = bin(int(hash_a, 16))[2:]
    bin_b = bin(int(hash_b, 16))[2:]
    max_len = max(len(bin_a), len(bin_b))
    bin_a = bin_a.zfill(max_len)
    bin_b = bin_b.zfill(max_len)
    return sum(a != b for a, b in zip(bin_a, bin_b))


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text normalisation for NLP models.

    Replaces @-mentions and URLs with placeholder tokens
    (aligned with dreamsApp/app/utils/sentiment.py ``preprocess``).
    """
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+", "http", text)
    return text.strip()


# ── Timestamp parsing ─────────────────────────────────────────────────────────

_TS_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",   # with microseconds + timezone (e.g. 2026-03-01T11:06:00.327879+00:00)
    "%Y-%m-%dT%H:%M:%S%z",      # with timezone, no microseconds
    "%Y-%m-%dT%H:%M:%S.%f",     # with microseconds, no timezone
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]


def parse_timestamp(ts: str | None) -> datetime | None:
    """Try multiple common ISO/CSV date formats and return a datetime or None.

    Handles microseconds, timezone offsets, and the ``Z`` suffix via
    ``datetime.fromisoformat()`` (Python 3.11+) as the first attempt,
    then falls back to explicit strptime patterns.
    """
    if not ts or not str(ts).strip():
        return None
    ts = str(ts).strip()

    # Fast path: fromisoformat handles virtually all ISO-8601 variants
    # including microseconds + timezone (e.g. 2026-03-01T10:31:11.429468+00:00)
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        pass

    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    logger.warning("Unparseable timestamp: %s", ts)
    return None


# ── Safe type casting ─────────────────────────────────────────────────────────

def safe_float(val: Any, default: float | None = None) -> float | None:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def safe_int(val: Any, default: int | None = None) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ── Unique ID generation ─────────────────────────────────────────────────────

def make_memory_id(user_id: str, image_filename: str) -> str:
    """Deterministic ID so that re-importing the same record is idempotent."""
    key = f"{user_id}::{image_filename}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
