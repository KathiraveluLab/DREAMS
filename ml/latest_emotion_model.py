"""
Latest Model Integration for Emotion Detection

This module wraps the face_classification model from latest-model directory
to provide emotion detection capabilities integrated with the DREAMS analytics server.

EMOTION LABELS (fer2013):
- angry, disgust, fear, happy, sad, surprise, neutral

This maps to the three-class system (positive/neutral/negative) for DREAMS.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import sys

# Add the latest-model/face_classification/src to path for imports
LATEST_MODEL_PATH = Path(__file__).parent.parent / "latest-model" / "face_classification"
sys.path.insert(0, str(LATEST_MODEL_PATH / "src"))

# Model paths 
EMOTION_MODEL_PATH = LATEST_MODEL_PATH / "trained_models" / "fer2013_mini_XCEPTION.119-0.65.hdf5"
DETECTION_MODEL_PATH = LATEST_MODEL_PATH / "trained_models" / "detection_models" / "facial-expression.xml"

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

# Global model cache to avoid reloading
_emotion_classifier = None
_face_detection = None


def _load_models():
    global _emotion_classifier, _face_detection
    
    if _emotion_classifier is None:
        try:
            from keras.models import load_model
            import cv2
            
            _emotion_classifier = load_model(str(EMOTION_MODEL_PATH), compile=False)
            _face_detection = cv2.CascadeClassifier(str(DETECTION_MODEL_PATH))
            print(f"[LatestModel] Loaded emotion model from {EMOTION_MODEL_PATH}")
        except Exception as e:
            print(f"[LatestModel] Error loading models: {e}")
            raise
    
    return _emotion_classifier, _face_detection


def _preprocess_face(gray_face: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    import cv2
    gray_face = cv2.resize(gray_face, target_size)
    gray_face = gray_face.astype('float32')
    gray_face = gray_face / 255.0
    gray_face = gray_face - 0.5
    gray_face = gray_face * 2.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    return gray_face


def detect_and_classify_emotion(image: np.ndarray) -> Dict:
    import cv2
    
    try:
        emotion_classifier, face_detection = _load_models()
    except Exception as e:
        return {
            "error": str(e),
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "uncertainty_margin": 0.25,
            "detected_facial_cues": [],
            "cue_confidence": 0.0,
            "notes": "Model loading failed. Defaulting to neutral.",
            "disclaimer": "Emotion estimated from facial cues only. Neutral is a valid state.",
        }
    
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    image_uint8 = (image * 255).astype(np.uint8)
    
    if len(image_uint8.shape) == 3:
        gray_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_uint8
    
    faces = face_detection.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    face_detected = len(faces) > 0
    
    if face_detected:
        face_areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(face_areas)
        x, y, w, h = faces[largest_idx]
        gray_face = gray_image[y:y+h, x:x+w]
        face_location = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
    else:
        gray_face = gray_image
        h, w = gray_image.shape[:2]
        face_location = {"x": 0, "y": 0, "width": int(w), "height": int(h)}
    
    target_size = emotion_classifier.input_shape[1:3]
    processed_face = _preprocess_face(gray_face, target_size)
    emotion_predictions = emotion_classifier.predict(processed_face, verbose=0)[0]
    
    detailed_emotions = {}
    for idx, prob in enumerate(emotion_predictions):
        label = FER2013_LABELS[idx]
        detailed_emotions[label] = float(prob)
    
    positive_prob = 0.0
    neutral_prob = 0.0
    negative_prob = 0.0
    
    for label, prob in detailed_emotions.items():
        category = EMOTION_CATEGORY_MAP[label]
        if category == 'positive':
            positive_prob += prob
        elif category == 'neutral':
            neutral_prob += prob
        else:
            negative_prob += prob
    
    dominant_idx = np.argmax(emotion_predictions)
    dominant_emotion = FER2013_LABELS[dominant_idx]
    dominant_confidence = float(emotion_predictions[dominant_idx])
    
    uncertainty = min(0.25, 0.05 + 0.2 * (1.0 - dominant_confidence))
    
    detected_cues = [dominant_emotion]
    face_note = "" if face_detected else " (no face bounding box detected, analyzed full image)"
    if dominant_confidence > 0.5:
        notes = f"Strong {dominant_emotion} expression detected ({dominant_confidence:.1%} confidence){face_note}."
    elif dominant_confidence > 0.3:
        notes = f"Moderate {dominant_emotion} expression detected ({dominant_confidence:.1%} confidence){face_note}."
    else:
        notes = f"Weak facial expression. {dominant_emotion} is most likely ({dominant_confidence:.1%}){face_note}."
    
    return {
        "positive": float(round(positive_prob, 4)),
        "neutral": float(round(neutral_prob, 4)),
        "negative": float(round(negative_prob, 4)),
        "uncertainty_margin": float(round(uncertainty, 4)),
        "detailed_emotions": detailed_emotions,
        "dominant_emotion": dominant_emotion,
        "dominant_confidence": float(round(dominant_confidence, 4)),
        "detected_facial_cues": detected_cues,
        "cue_confidence": float(round(dominant_confidence, 4)),
        "notes": notes,
        "disclaimer": "Emotion estimated using fer2013_mini_XCEPTION CNN model. Results are probabilistic.",
        "model": "fer2013_mini_XCEPTION",
        "face_detected": face_detected,
        "face_location": face_location,
    }


def estimate_emotion_from_image(image: np.ndarray) -> Dict:
    return detect_and_classify_emotion(image)


def compare_image_estimates(image_a: np.ndarray, image_b: np.ndarray) -> Dict:
    estimate_a = estimate_emotion_from_image(image_a)
    estimate_b = estimate_emotion_from_image(image_b)
    
    prob_diff = {
        "positive": abs(estimate_a["positive"] - estimate_b["positive"]),
        "neutral": abs(estimate_a["neutral"] - estimate_b["neutral"]),
        "negative": abs(estimate_a["negative"] - estimate_b["negative"]),
    }
    
    return {
        "image_a": estimate_a,
        "image_b": estimate_b,
        "probability_differences": prob_diff,
        "different_distributions": any(d > 0.01 for d in prob_diff.values()),
        "note": "Emotions derived using CNN model trained on fer2013 dataset.",
    }
