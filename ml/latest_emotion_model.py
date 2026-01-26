"""
Latest Model Integration for Emotion Detection

This module wraps the face_classification model from latest-model directory
to provide emotion detection capabilities integrated with the DREAMS analytics server.

The fer2013_mini_XCEPTION Keras model requires TensorFlow which doesn't support
Python 3.14. This module uses a subprocess call to Python 3.11 to run inference.

EMOTION LABELS (fer2013):
- angry, disgust, fear, happy, sad, surprise, neutral

This maps to the three-class system (positive/neutral/negative) for DREAMS.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict
import numpy as np

# Path to the inference script
INFERENCE_SCRIPT = Path(__file__).parent / "keras_inference.py"
PYTHON_311_PATH = __import__('os').environ.get("PYTHON_311_PATH", "python3.11")

# Emotion labels from fer2013 dataset
FER2013_LABELS = {
    0: 'angry', 
    1: 'disgust', 
    2: 'fear', 
    3: 'happy',
    4: 'sad', 
    5: 'surprise', 
    6: 'neutral'
}

# Mapping to three-class system for DREAMS
EMOTION_CATEGORY_MAP = {
    'angry': 'negative',
    'disgust': 'negative',
    'fear': 'negative',
    'sad': 'negative',
    'happy': 'positive',
    'surprise': 'positive',
    'neutral': 'neutral'
}


def _run_inference_subprocess(image_path: str) -> Dict:
    """Run the Keras model via subprocess using Python 3.11."""
    try:
        result = subprocess.run(
            [PYTHON_311_PATH, str(INFERENCE_SCRIPT), image_path],
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **dict(__import__('os').environ),
                'TF_CPP_MIN_LOG_LEVEL': '2'
            }
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return {
                "error": f"Inference failed: {error_msg}",
                "positive": 0.0,
                "neutral": 1.0,
                "negative": 0.0,
                "uncertainty_margin": 0.25,
                "notes": "Model inference failed. Defaulting to neutral.",
            }
        
        return json.loads(result.stdout)
        
    except subprocess.TimeoutExpired:
        return {
            "error": "Inference timeout",
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "uncertainty_margin": 0.25,
            "notes": "Model inference timed out. Defaulting to neutral.",
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON response: {e}",
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "uncertainty_margin": 0.25,
            "notes": "Model returned invalid response. Defaulting to neutral.",
        }
    except Exception as e:
        return {
            "error": str(e),
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "uncertainty_margin": 0.25,
            "notes": f"Error: {e}. Defaulting to neutral.",
        }


def detect_and_classify_emotion(image: np.ndarray) -> Dict:
    """
    Detect and classify emotions in an image.
    
    This function saves the image to a temp file and calls the Python 3.11
    subprocess to run inference using the Keras model from latest-model.
    
    Args:
        image: RGB image as numpy array (H, W, 3), values 0-1 or 0-255
    
    Returns:
        Dict with positive/neutral/negative probabilities and metadata
    """
    import tempfile
    from PIL import Image
    
    try:
        # Convert numpy array to PIL and save to temp file
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        pil_image = Image.fromarray(image_uint8)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path)
        
        # Run inference
        result = _run_inference_subprocess(tmp_path)
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "uncertainty_margin": 0.25,
            "detected_facial_cues": [],
            "cue_confidence": 0.0,
            "notes": f"Error processing image: {e}. Defaulting to neutral.",
            "disclaimer": "Emotion estimated from facial cues only. Neutral is a valid state.",
        }


def estimate_emotion_from_image(image: np.ndarray) -> Dict:
    """Wrapper for detect_and_classify_emotion."""
    return detect_and_classify_emotion(image)


def compare_image_estimates(image_a: np.ndarray, image_b: np.ndarray) -> Dict:
    """Compare emotion estimates between two images."""
    estimate_a = estimate_emotion_from_image(image_a)
    estimate_b = estimate_emotion_from_image(image_b)
    
    prob_diff = {
        "positive": abs(estimate_a.get("positive", 0) - estimate_b.get("positive", 0)),
        "neutral": abs(estimate_a.get("neutral", 0) - estimate_b.get("neutral", 0)),
        "negative": abs(estimate_a.get("negative", 0) - estimate_b.get("negative", 0)),
    }
    
    return {
        "image_a": estimate_a,
        "image_b": estimate_b,
        "probability_differences": prob_diff,
        "different_distributions": any(d > 0.01 for d in prob_diff.values()),
        "note": "Emotions derived using fer2013_mini_XCEPTION CNN model from latest-model directory.",
    }
