"""
Step 5 — Emotions: Extract rich, multi-model emotion signals.

Key improvement over prior pipeline:
- 7 discrete emotions (anger, disgust, fear, joy, neutral, sadness, surprise)
- Valence / arousal continuous scores
- Positive / negative / neutral sentiment
- CHIME recovery category — **directly reuses** dreamsApp's
  ``SentimentAnalyzer`` (via ``analysis_pipeline.compat``) including
  its federated-learning self-correction logic.
- All from a single pass, stored together in emotion_scores table

CHIME integration
-----------------
Instead of reimplementing the CHIME pipeline, this step imports
``chime_analyzer`` from ``analysis_pipeline.compat``, which is a
``PipelineSentimentAnalyzer`` — a subclass of the *real*
``dreamsApp/app/utils/sentiment.SentimentAnalyzer``.  The only
override is ``get_chime_classifier()`` which resolves the local
federated-learning model path without Flask's ``current_app``.
``analyze_chime()`` is inherited unchanged.

If dreamsApp's sentiment module cannot be loaded (e.g. Flask not
installed), CHIME classification is skipped gracefully.

Memory-efficient design
-----------------------
Each of the 4 sub-models is loaded **one at a time**, run across
all pending records, then deleted + gc.collect() before the next
model is loaded.  This keeps peak RAM ≈ 500 MB instead of ≈ 1.6 GB.
"""

import gc
import logging

from ..config import (
    DISCRETE_EMOTION_MODEL, VA_EMOTION_MODEL,
    SENTIMENT_MODEL_NAME, BATCH_SIZE,
)
from ..db import (
    get_db, get_pending_ids,
    mark_record_done, mark_record_error,
)
from ..utils import clean_text
from ..compat import chime_analyzer, DREAMSAPP_CHIME_AVAILABLE

logger = logging.getLogger(__name__)

_MAX_INPUT_LEN = 512


def _derive_valence_arousal(discrete: dict) -> tuple[float, float]:
    """Derive valence and arousal from Ekman discrete emotions.

    Used as a fallback when the dedicated VA model is unavailable or
    returns null.  Mappings are grounded in Russell's circumplex model:
    - Valence:  joy is positive; fear, sadness, anger, disgust are negative.
    - Arousal:  fear, anger, surprise are activating;
                sadness and neutral are deactivating.

    Returns values clamped to [-1.0, 1.0].
    """
    joy      = discrete.get("joy",      0.0)
    fear     = discrete.get("fear",     0.0)
    sadness  = discrete.get("sadness",  0.0)
    anger    = discrete.get("anger",    0.0)
    disgust  = discrete.get("disgust",  0.0)
    surprise = discrete.get("surprise", 0.0)
    neutral  = discrete.get("neutral",  0.0)

    valence = joy - (fear + sadness + anger + disgust) / 4.0
    arousal = (fear + anger + surprise) / 3.0 - (sadness + neutral) / 2.0

    return (
        max(-1.0, min(1.0, round(valence, 4))),
        max(-1.0, min(1.0, round(arousal, 4))),
    )


def _unload(*objs) -> None:
    """Delete objects and free memory."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def run(log: logging.Logger | None = None) -> int:
    """Run multi-model emotion extraction on all pending records."""
    _log = log or logger

    from transformers import pipeline as hf_pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    device_label = "cuda" if device == 0 else "cpu"

    # ── Gather pending records ───────────────────────────────────────────────
    pending = get_pending_ids("emotions")
    if not pending:
        _log.info("All records already have emotion scores.")
        return 0

    conn = get_db()
    try:
        placeholders = ",".join("?" for _ in pending)
        rows = conn.execute(
            f"""SELECT memory_id, caption, generated_caption
                FROM memories WHERE memory_id IN ({placeholders})""",
            pending,
        ).fetchall()
    finally:
        conn.close()

    # Pre-clean all texts
    texts_by_id: dict[str, str] = {}
    for row in rows:
        text = row["caption"] or row["generated_caption"]
        if text and text.strip():
            texts_by_id[row["memory_id"]] = clean_text(text)[:_MAX_INPUT_LEN]

    if not texts_by_id:
        _log.info("No text available for emotion analysis.")
        return 0

    # Accumulators: emotion results per memory_id
    emo_data: dict[str, dict] = {mid: {} for mid in texts_by_id}
    mids = list(texts_by_id.keys())
    texts = list(texts_by_id.values())

    _log.info("Extracting emotions for %d records on %s...", len(texts_by_id), device_label)

    # ── Sub-model 1: Discrete emotions (≈330 MB) ────────────────────────────
    _log.info("Loading discrete-emotion model: %s", DISCRETE_EMOTION_MODEL)
    try:
        emotion_pipe = hf_pipeline(
            "text-classification",
            model=DISCRETE_EMOTION_MODEL,
            top_k=None,
            device=device,
        )
        
        batch_out = emotion_pipe(texts, batch_size=BATCH_SIZE)
        for mid, results in zip(mids, batch_out):
            try:
                scores = {e["label"]: float(e["score"]) for e in results}
                emo_data[mid]["discrete"] = scores
                emo_data[mid]["dominant"] = max(scores, key=scores.get)
            except Exception as e:
                _log.warning("Discrete emotion parsing failed for %s: %s", mid, e)
        
    except Exception as exc:
        _log.warning("Discrete emotion model failed to load or run: %s", exc)
    finally :
        if emotion_pipe:
            del emotion_pipe
    
    _unload()
    _log.info("Discrete emotions done — model unloaded.")

    # ── Sub-model 2: Valence / Arousal (≈330 MB) ────────────────────────────
    _log.info("Loading valence-arousal model: %s", VA_EMOTION_MODEL)
    try:
        va_pipe = hf_pipeline(
            "text-classification",
            model=VA_EMOTION_MODEL,
            top_k=None,
            device=device,
        )
       
        batch_out = va_pipe(texts, batch_size=BATCH_SIZE)
        for mid, results in zip(mids, batch_out):
            try:
                # Handle potential list nesting in some HF models
                va_items = results[0] if isinstance(results[0], list) else results
                va_dict = {r["label"]: float(r["score"]) for r in va_items}
                emo_data[mid]["valence"] = va_dict.get("valence") or va_dict.get("Valence")
                emo_data[mid]["arousal"] = va_dict.get("arousal") or va_dict.get("Arousal")
            except Exception as e:
                _log.warning("Valence/arousal parsing failed for %s: %s", mid, e)
        del va_pipe
    except Exception as exc:
        _log.warning("Valence-arousal model unavailable (%s); skipping.", exc)
    _unload()
    _log.info("Valence-arousal done — model unloaded.")

    # ── Sub-model 3: Sentiment (≈500 MB) ────────────────────────────────────
    _log.info("Loading sentiment model: %s", SENTIMENT_MODEL_NAME)
    try:
        sent_pipe = hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
            top_k=None,
            device=device,
        )
        
        batch_out = sent_pipe(texts, batch_size=BATCH_SIZE)
        for mid, results in zip(mids, batch_out):
            try:
                sent_dict = {r["label"].lower(): float(r["score"]) for r in results}
                emo_data[mid]["sent_pos"] = sent_dict.get("positive", sent_dict.get("pos"))
                emo_data[mid]["sent_neg"] = sent_dict.get("negative", sent_dict.get("neg"))
                emo_data[mid]["sent_neu"] = sent_dict.get("neutral", sent_dict.get("neu"))
                emo_data[mid]["sent_label"] = max(sent_dict, key=sent_dict.get)
            except Exception:
                pass
        del sent_pipe
    except Exception as exc:
        _log.warning("Sentiment model unavailable (%s); skipping.", exc)
    _unload()
    _log.info("Sentiment done — model unloaded.")

    # ── Sub-model 4: CHIME recovery (≈ 440 MB) ─────────────────────────────
    if DREAMSAPP_CHIME_AVAILABLE:
        _log.info("CHIME: using dreamsApp SentimentAnalyzer (with FL self-correction)")
        classifier = chime_analyzer.get_chime_classifier()
        if not classifier:
            _log.warning("CHIME classifier failed to load; CHIME scores will be NULL.")
        else:
            try:
                # Vectorized inference directly on underlying pipeline
                batch_out = classifier(texts, batch_size=BATCH_SIZE)
                for mid, results in zip(mids, batch_out):
                    try:
                        # Find the highest scoring category
                        top_result = max(results, key=lambda x: x['score'])
                        emo_data[mid]["chime_cat"] = top_result["label"]
                        emo_data[mid]["chime_conf"] = float(top_result["score"])
                    except Exception as e:
                        _log.warning("CHIME result parsing failed for %s: %s", mid, e)
            except Exception as e:
                _log.warning("CHIME inference batch failed: %s", e)
        # Clear CHIME model references
        try:
            chime_analyzer._chime_classifier = None
        except Exception:
            pass
        _unload()
        _log.info("CHIME done — model unloaded.")
    else:
        _log.warning("dreamsApp sentiment module not available; CHIME scores will be NULL.")

    # ── Fallback: derive valence/arousal from discrete if VA model gave null ──
    _log.info("Applying valence/arousal fallback for records where VA model returned null …")
    fallback_count = 0
    for mid, data in emo_data.items():
        if data.get("valence") is None or data.get("arousal") is None:
            discrete = data.get("discrete", {})
            if discrete:
                v, a = _derive_valence_arousal(discrete)
                if data.get("valence") is None:
                    data["valence"] = v
                if data.get("arousal") is None:
                    data["arousal"] = a
                fallback_count += 1
    if fallback_count:
        _log.info("  Derived valence/arousal for %d record(s) via circumplex fallback.", fallback_count)

    # ── Write all results to DB ──────────────────────────────────────────────
    conn = get_db()
    try:
        processed = 0
        for mid, data in emo_data.items():
            discrete = data.get("discrete", {})
            if not discrete:
                mark_record_error(mid, "emotions", "Discrete emotion extraction failed", conn)
                continue
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO emotion_scores
                       (memory_id,
                        anger, disgust, fear, joy, neutral, sadness, surprise,
                        dominant_emotion,
                        valence, arousal,
                        sentiment_label, sentiment_pos, sentiment_neg, sentiment_neu,
                        chime_category, chime_confidence)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mid,
                        discrete.get("anger"),
                        discrete.get("disgust"),
                        discrete.get("fear"),
                        discrete.get("joy"),
                        discrete.get("neutral"),
                        discrete.get("sadness"),
                        discrete.get("surprise"),
                        data.get("dominant"),
                        data.get("valence"),
                        data.get("arousal"),
                        data.get("sent_label"),
                        data.get("sent_pos"),
                        data.get("sent_neg"),
                        data.get("sent_neu"),
                        data.get("chime_cat"),
                        data.get("chime_conf"),
                    ),
                )
                mark_record_done(mid, "emotions", conn)
                processed += 1
            except Exception as e:
                _log.warning("Emotion DB write failed for %s: %s", mid, e)
                mark_record_error(mid, "emotions", str(e), conn)

        conn.commit()
        _log.info("Emotion extraction complete: %d processed.", processed)
        return processed
    finally:
        conn.close()
