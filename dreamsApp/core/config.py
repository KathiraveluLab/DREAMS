"""
dreamsApp.core.config
~~~~~~~~~~~~~~~~~~~~~
PipelineConfig — the single source of truth for all DREAMS algorithm parameters.

Usage (in code)::

    from dreamsApp.core.config import PipelineConfig

    config = PipelineConfig()                        # all defaults
    config = PipelineConfig(gap_threshold_hours=48)  # custom
    config = PipelineConfig.from_yaml("my_experiment.yaml")

Usage (YAML-driven experiment)::

    # Copy config_defaults.yaml, edit, then load:
    config = PipelineConfig.from_yaml("experiment_01.yaml")
    pipeline = DreamsPipeline(config=config)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable configuration for the DREAMS pipeline.

    All parameters have research-friendly defaults that match the original
    hard-coded values, so existing code keeps working without changes.

    Attributes
    ----------
    gap_threshold_hours:
        Minimum silence gap (in hours) between emotion events that causes a new
        Episode to be created during timeline segmentation.
    adjacency_threshold_days:
        Maximum temporal distance (in days) between two Episodes for them to be
        considered *adjacent* (rather than *disjoint*) when building graph edges.

    sentiment_model_id:
        HuggingFace model ID (or local path) for text sentiment classification.
    text_embedding_model_id:
        SentenceTransformer model ID for dense text encoding (stored in ChromaDB).
    image_embedding_model_id:
        SentenceTransformer CLIP model ID for dense image encoding (stored in ChromaDB).

    enable_image_embedding:
        When True, CLIP image embeddings are computed during analyze_entry().
        Set False to skip image encoding (e.g. text-only research datasets).
    enable_location_extraction:
        When True, GPS EXIF data is extracted from images during analyze_entry().
    """

    # ── Graph / segmentation ───────────────────────────────────────────────────
    gap_threshold_hours: int = 24
    adjacency_threshold_days: int = 7

    # ── Active model IDs ───────────────────────────────────────────────────────
    sentiment_model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    text_embedding_model_id: str = "all-MiniLM-L6-v2"
    image_embedding_model_id: str = "clip-ViT-B-32"

    # ── Feature flags ──────────────────────────────────────────────────────────
    enable_image_embedding: bool = True
    enable_location_extraction: bool = True

    # ──────────────────────────────────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """
        Build a PipelineConfig from a plain dictionary.

        Unknown keys are silently ignored so that partial overrides work::

            PipelineConfig.from_dict({"gap_threshold_hours": 48})
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """
        Load a PipelineConfig from a YAML file.

        The YAML file should contain a flat mapping of parameter names to values.
        See ``config_defaults.yaml`` for the canonical example.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If PyYAML (``pyyaml``) is not installed.
        """
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load config from YAML. "
                "Install it with: pip install pyyaml"
            ) from exc

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return cls.from_dict(data)

    # ──────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return all parameters as a plain dictionary."""
        return asdict(self)


