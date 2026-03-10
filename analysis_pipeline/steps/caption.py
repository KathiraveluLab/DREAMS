"""
Step 2 — Caption: Generate captions for images that don't already have one.

Uses BLIP (already in dreamsApp) to produce natural-language descriptions
of photos.  Stores the generated caption in memories.generated_caption,
preserving the original user caption in memories.caption.
"""

import gc
import logging
from pathlib import Path

from ..config import BLIP_MODEL_NAME, BATCH_SIZE, PROCESSED_DIR, DATA_DIR
from ..db import get_db, get_pending_ids, mark_record_done, mark_record_error
from ..utils import validate_safe_path

logger = logging.getLogger(__name__)


def run(log: logging.Logger | None = None) -> int:
    """Generate BLIP captions for images that lack a generated_caption.

    If the user already provided a caption, BLIP is **not loaded at all**.
    The user caption is copied to ``generated_caption`` so downstream
    steps always have text to work with.  BLIP is only downloaded and
    loaded for images that have *no* caption whatsoever.
    """
    _log = log or logger

    conn = get_db()
    try:
        # ── Find records needing a generated caption ─────────────────────
        rows = conn.execute(
            """SELECT memory_id, image_path, caption
               FROM memories
               WHERE is_duplicate = 0
                 AND image_path IS NOT NULL
                 AND generated_caption IS NULL"""
        ).fetchall()

        if not rows:
            _log.info("No images need captioning.")
            return 0

        # Split: records that already have a user caption vs those that don't
        has_caption = [r for r in rows if r["caption"] and r["caption"].strip()]
        needs_blip  = [r for r in rows if not r["caption"] or not r["caption"].strip()]

        processed = 0

        # ── Fast path: copy user caption → generated_caption (no ML) ─────
        for row in has_caption:
            mid = row["memory_id"]
            conn.execute(
                "UPDATE memories SET generated_caption=? WHERE memory_id=?",
                (row["caption"].strip(), mid),
            )
            mark_record_done(mid, "caption", conn)
            processed += 1
        if has_caption:
            conn.commit()
            _log.info(
                "Copied %d user-provided caption(s) — BLIP not needed.",
                len(has_caption),
            )

        # ── Slow path: generate caption with BLIP (only if required) ─────
        if needs_blip:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image

            _log.info(
                "Loading BLIP model for %d image(s) without caption: %s",
                len(needs_blip), BLIP_MODEL_NAME,
            )
            processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
            model.eval()
            _log.info("BLIP model loaded.")

            for i in range(0, len(needs_blip), BATCH_SIZE):
                batch = needs_blip[i : i + BATCH_SIZE]
                for row in batch:
                    mid = row["memory_id"]
                    img_path = row["image_path"]
                    try:
                        if not img_path:
                            mark_record_error(mid, "caption", "No image path", conn)
                            continue
                        try:
                            safe_path = validate_safe_path(
                                img_path, allowed_roots=[PROCESSED_DIR, DATA_DIR],
                            )
                        except ValueError:
                            mark_record_error(mid, "caption", "Image path outside allowed dirs", conn)
                            continue
                        if not safe_path.exists():
                            mark_record_error(mid, "caption", "Image file not found", conn)
                            continue

                        img = Image.open(str(safe_path)).convert("RGB")
                        inputs = processor(img, return_tensors="pt")
                        output_ids = model.generate(**inputs, max_new_tokens=80)
                        generated = processor.decode(
                            output_ids[0], skip_special_tokens=True,
                        ).strip()

                        conn.execute(
                            "UPDATE memories SET generated_caption=? WHERE memory_id=?",
                            (generated, mid),
                        )
                        mark_record_done(mid, "caption", conn)
                        processed += 1
                    except Exception as e:
                        _log.warning("Caption failed for %s: %s", mid, e)
                        mark_record_error(mid, "caption", str(e), conn)

                conn.commit()
                _log.info(
                    "  Captioned %d / %d",
                    min(i + BATCH_SIZE, len(needs_blip)), len(needs_blip),
                )

            # Free BLIP from RAM
            del model, processor
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            _log.info("BLIP model unloaded.")

        _log.info("Captioning complete: %d processed.", processed)
        return processed
    finally:
        conn.close()
