import logging
import threading
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Cache models to avoid reloading on every ingestion
_text_embedding_model = None
_text_embedding_model_id = None
_text_model_lock = threading.Lock()

_image_embedding_model = None
_image_embedding_model_id = None
_image_model_lock = threading.Lock()

def _get_text_model(model_id: str):
    global _text_embedding_model, _text_embedding_model_id
    if _text_embedding_model is None or _text_embedding_model_id != model_id:
        with _text_model_lock:
            if _text_embedding_model is None or _text_embedding_model_id != model_id:
                _text_embedding_model = SentenceTransformer(model_id)
                _text_embedding_model_id = model_id
    return _text_embedding_model

def _get_image_model(model_id: str):
    global _image_embedding_model, _image_embedding_model_id
    if _image_embedding_model is None or _image_embedding_model_id != model_id:
        with _image_model_lock:
            if _image_embedding_model is None or _image_embedding_model_id != model_id:
                _image_embedding_model = SentenceTransformer(model_id)
                _image_embedding_model_id = model_id
    return _image_embedding_model

def get_text_embedding(text: str, model_id: str):
    """Generates dense vectors for textual inputs."""
    if not text or not text.strip():
        return None
    model = _get_text_model(model_id)
    return model.encode(text).tolist()

def get_image_embedding(image_path: str, model_id: str):
    """Generates dense vectors for vision inputs using CLIP/ViT."""
    try:
        model = _get_image_model(model_id)
        with Image.open(image_path) as img:
            return model.encode(img).tolist()
    except Exception as e:
        logger.error(f"Failed to generate CLIP embedding for {image_path}: {e}")
        return None
