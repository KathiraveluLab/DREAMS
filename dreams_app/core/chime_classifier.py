import os


def resolve_chime_model_path(default_model_id: str, logger):
    """Return local FL model path when available, else default model id."""
    model_path = default_model_id

    try:
        from flask import has_app_context, current_app
        if has_app_context():
            local_model_path = os.path.join(
                current_app.root_path,
                "models",
                "production_chime_model",
            )
            if os.path.exists(local_model_path):
                logger.info(
                    ">>> SELF-CORRECTION: Learned model found at %s. Loading...",
                    local_model_path,
                )
                model_path = local_model_path
    except RuntimeError:
        # No active Flask app context (e.g., tests/CLI)
        pass

    return model_path


def init_chime_classifier(current_classifier, pipeline_fn, default_model_id: str, logger):
    """Create CHIME classifier once and return cached instance."""
    if current_classifier is not None:
        return current_classifier

    try:
        if pipeline_fn is None:
            raise RuntimeError("transformers is required for CHIME inference")

        model_path = resolve_chime_model_path(default_model_id, logger)
        return pipeline_fn(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            top_k=None,
        )
    except Exception as e:
        logger.error("Error loading CHIME model: %s", e)
        return None


def pick_top_chime_result(results):
    """Handle both [dict] and [[dict]] outputs and return top score item."""
    if not results:
        return {"label": "Uncategorized", "score": 0.0}

    if isinstance(results[0], list):
        if not results[0]:
            return {"label": "Uncategorized", "score": 0.0}
        return max(results[0], key=lambda x: x.get("score", 0.0))

    if isinstance(results[0], dict):
        return max(results, key=lambda x: x.get("score", 0.0))

    return {"label": "Uncategorized", "score": 0.0}
