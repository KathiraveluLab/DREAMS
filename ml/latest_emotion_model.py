"""
Latest Model Integration for Emotion Detection

This module wraps the face_classification model from latest-model directory
to provide emotion detection capabilities integrated with the DREAMS analytics server.

The fer2013_mini_XCEPTION Keras model requires TensorFlow which doesn't support
Python 3.14. This module uses a subprocess call to Python 3.11 to run inference.

EMOTION LABELS (6 Basic Emotions):
- Happiness, Sadness, Fear, Anger, Disgust, Surprise

This uses the 6 basic emotions model (Ekman's model) for DREAMS.
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

# Emotion labels from fer2013 dataset (mapped to 6 basic emotions)
FER2013_LABELS = {
    0: 'Anger', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happiness',
    4: 'Sadness', 
    5: 'Surprise', 
    6: 'Surprise'  # neutral mapped to Surprise as closest match
}

# 6 Basic Emotions for DREAMS
BASIC_EMOTIONS = ['Happiness', 'Sadness', 'Fear', 'Anger', 'Disgust', 'Surprise']


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
                "Happiness": 0.0,
                "Sadness": 0.0,
                "Fear": 0.0,
                "Anger": 0.0,
                "Disgust": 0.0,
                "Surprise": 0.0,
                "uncertainty_margin": 0.25,
                "notes": "Model inference failed.",
            }
        
        return json.loads(result.stdout)
        
    except subprocess.TimeoutExpired:
        return {
            "error": "Inference timeout",
            "Happiness": 0.0,
            "Sadness": 0.0,
            "Fear": 0.0,
            "Anger": 0.0,
            "Disgust": 0.0,
            "Surprise": 0.0,
            "uncertainty_margin": 0.25,
            "notes": "Model inference timed out.",
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON response: {e}",
            "Happiness": 0.0,
            "Sadness": 0.0,
            "Fear": 0.0,
            "Anger": 0.0,
            "Disgust": 0.0,
            "Surprise": 0.0,
            "uncertainty_margin": 0.25,
            "notes": "Model returned invalid response.",
        }
    except Exception as e:
        return {
            "error": str(e),
            "Happiness": 0.0,
            "Sadness": 0.0,
            "Fear": 0.0,
            "Anger": 0.0,
            "Disgust": 0.0,
            "Surprise": 0.0,
            "uncertainty_margin": 0.25,
            "notes": f"Error: {e}.",
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
            "Happiness": 0.0,
            "Sadness": 0.0,
            "Fear": 0.0,
            "Anger": 0.0,
            "Disgust": 0.0,
            "Surprise": 0.0,
            "uncertainty_margin": 0.25,
            "detected_facial_cues": [],
            "cue_confidence": 0.0,
            "notes": f"Error processing image: {e}.",
            "disclaimer": "Emotion estimated from facial cues only using 6 basic emotions.",
        }


def estimate_emotion_from_image(image: np.ndarray) -> Dict:
    """Wrapper for detect_and_classify_emotion."""
    return detect_and_classify_emotion(image)


def compare_image_estimates(image_a: np.ndarray, image_b: np.ndarray) -> Dict:
    """Compare emotion estimates between two images."""
    estimate_a = estimate_emotion_from_image(image_a)
    estimate_b = estimate_emotion_from_image(image_b)
    
    prob_diff = {
        "Happiness": abs(estimate_a.get("Happiness", 0) - estimate_b.get("Happiness", 0)),
        "Sadness": abs(estimate_a.get("Sadness", 0) - estimate_b.get("Sadness", 0)),
        "Fear": abs(estimate_a.get("Fear", 0) - estimate_b.get("Fear", 0)),
        "Anger": abs(estimate_a.get("Anger", 0) - estimate_b.get("Anger", 0)),
        "Disgust": abs(estimate_a.get("Disgust", 0) - estimate_b.get("Disgust", 0)),
        "Surprise": abs(estimate_a.get("Surprise", 0) - estimate_b.get("Surprise", 0)),
    }
    
    return {
        "image_a": estimate_a,
        "image_b": estimate_b,
        "probability_differences": prob_diff,
        "different_distributions": any(d > 0.01 for d in prob_diff.values()),
        "note": "Emotions derived using fer2013_mini_XCEPTION CNN model with 6 basic emotions.",
    }
