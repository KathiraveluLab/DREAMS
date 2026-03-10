"""
Compatibility bridge — direct reuse of dreamsApp shared modules.

This module loads ``dreamsApp/app/utils/sentiment.py`` via ``importlib``
(because ``dreamsApp/app/utils/`` has no ``__init__.py``) and provides a
``PipelineSentimentAnalyzer`` that **inherits** the real
``SentimentAnalyzer`` — identical model-loading, identical inference,
identical federated-learning self-correction — but resolves the local
CHIME model path without ``flask.current_app``.

What is directly reused (real class, not mirrored):
* ``SentimentAnalyzer.analyze_chime()``    — inherited as-is
* ``SentimentAnalyzer.get_sentiment_models()``  — inherited as-is
* ``SentimentAnalyzer.get_blip_models()``  — inherited as-is
* ``SentimentAnalyzer.get_chime_classifier()``  — **overridden** only to
  remove the ``flask.current_app`` dependency; the
  ``transformers.pipeline(...)`` call is identical.

What is re-exported (Flask-free analytics packages):
* ``EmotionEvent``, ``EmotionTimeline`` from ``dreamsApp.analytics``
* Proximity / windowing utilities from ``dreamsApp.analytics``

Why importlib?
  ``dreamsApp/app/utils/`` has no ``__init__.py`` and ``dreamsApp/``
  itself is not a proper Python package, so normal ``from dreamsApp.app...``
  imports fail.  ``importlib.util.spec_from_file_location`` sidesteps
  this by loading the ``.py`` file directly.

Future:
  If ``dreamsApp/app/utils/`` gains an ``__init__.py`` and
  ``get_chime_classifier`` is decoupled from Flask, this bridge can be
  replaced with a plain ``from dreamsApp.app.utils.sentiment import
  SentimentAnalyzer``.
"""

import sys
import os
import logging
import importlib.util
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_root_str = str(_PROJECT_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

# ── Re-export dreamsApp analytics (Flask-free) ────────────────────────────────

try:
    from dreamsApp.analytics.emotion_timeline import EmotionEvent, EmotionTimeline  # noqa: F401
    from dreamsApp.analytics.emotion_proximity import (                            # noqa: F401
        segment_timeline_into_windows,
        compare_timelines_distance,
        compute_timeline_self_similarity,
    )
    from dreamsApp.analytics.time_aware_proximity import (                         # noqa: F401
        align_timelines_by_window,
        temporal_distance,
        proximity_matrix,
    )

    DREAMSAPP_ANALYTICS_AVAILABLE = True
    logger.debug("dreamsApp analytics modules imported successfully.")
except ImportError as exc:
    DREAMSAPP_ANALYTICS_AVAILABLE = False
    logger.info(
        "dreamsApp analytics not importable (%s); "
        "Phase 2/3 modules will use standalone implementations.",
        exc,
    )

# ── Load SentimentAnalyzer via importlib ──────────────────────────────────────
#
# sentiment.py lives at dreamsApp/app/utils/sentiment.py but that
# directory is not a Python package.  We load it as a standalone module.

_SENTIMENT_FILE = _PROJECT_ROOT / "dreamsApp" / "app" / "utils" / "sentiment.py"
_sentiment_mod = None

if _SENTIMENT_FILE.exists():
    try:
        _spec = importlib.util.spec_from_file_location(
            "dreamsApp_sentiment", str(_SENTIMENT_FILE)
        )
        _sentiment_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_sentiment_mod)
        logger.debug("dreamsApp sentiment module loaded from %s", _SENTIMENT_FILE)
    except Exception as exc:
        logger.info("Could not load dreamsApp sentiment module: %s", exc)
        _sentiment_mod = None
else:
    logger.info("dreamsApp sentiment module not found at %s", _SENTIMENT_FILE)


# ── PipelineSentimentAnalyzer ─────────────────────────────────────────────────
#
# Subclass of the *real* SentimentAnalyzer.  Only get_chime_classifier()
# is overridden — everything else (analyze_chime, get_sentiment_models,
# get_blip_models, etc.) is inherited and runs the original code.

DREAMSAPP_CHIME_AVAILABLE = False
chime_analyzer = None  # singleton, set below if import succeeds

if _sentiment_mod is not None:
    _BaseSentimentAnalyzer = _sentiment_mod.SentimentAnalyzer
    _BASE_HF_MODEL_ID = _sentiment_mod.HF_MODEL_ID  # "ashh007/dreams-chime-bert"

    class PipelineSentimentAnalyzer(_BaseSentimentAnalyzer):
        """SentimentAnalyzer that resolves the FL model path without Flask.

        The *only* change from the parent class is replacing
        ``flask.current_app.root_path`` with a static file-system lookup
        under ``<PROJECT_ROOT>/dreamsApp/app/models/production_chime_model``.

        If a federated-learning fine-tuned model exists there, it is used
        (self-correction feature).  Otherwise the base HuggingFace model
        ``ashh007/dreams-chime-bert`` is loaded — identical to the original.
        """

        def get_chime_classifier(self):
            if self._chime_classifier is not None:
                return self._chime_classifier

            try:
                from transformers import pipeline as hf_pipeline

                # Same path the Flask app resolves via current_app.root_path:
                #   dreamsApp/app/models/production_chime_model
                local_model_path = str(
                    _PROJECT_ROOT / "dreamsApp" / "app" / "models" / "production_chime_model"
                )

                model_path = _BASE_HF_MODEL_ID

                if os.path.isdir(local_model_path):
                    logging.info(
                        ">>> SELF-CORRECTION: Learned model found at %s. Loading...",
                        local_model_path,
                    )
                    model_path = local_model_path
                else:
                    logging.info(
                        "Loading Base CHIME model from Hugging Face: %s...",
                        _BASE_HF_MODEL_ID,
                    )

                self._chime_classifier = hf_pipeline(
                    "text-classification",
                    model=model_path,
                    tokenizer=model_path,
                    return_all_scores=True,
                )
                logging.info("CHIME model loaded successfully.")
            except Exception as exc:
                logging.error("Error loading CHIME model: %s", exc)
                return None

            return self._chime_classifier

    chime_analyzer = PipelineSentimentAnalyzer()
    DREAMSAPP_CHIME_AVAILABLE = True
    logger.debug("PipelineSentimentAnalyzer ready (CHIME + sentiment + BLIP).")


# ── Constants (kept for reference / standalone fallback) ──────────────────────

CHIME_HF_MODEL_ID = "ashh007/dreams-chime-bert"
SENTIMENT_HF_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
