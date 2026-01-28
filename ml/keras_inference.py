#!/usr/bin/env python3.11
"""
Standalone Keras Model Inference Script

This script runs with Python 3.11 (which has TensorFlow support) and loads
the fer2013_mini_XCEPTION model from the latest-model directory.

Usage:
    python3.11 keras_inference.py <image_path>
    
Outputs JSON to stdout with emotion predictions.
"""

import json
import sys
import os
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR.parent / "latest-model" / "face_classification" / "trained_models" / "fer2013_mini_XCEPTION.119-0.65.hdf5"
DETECTION_MODEL_PATH = SCRIPT_DIR.parent / "latest-model" / "face_classification" / "trained_models" / "detection_models" / "facial-expression.xml"

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


def preprocess_face(gray_face, target_size=(64, 64)):
    """Preprocess face image for the model."""
    import cv2
    import numpy as np
    
    gray_face = cv2.resize(gray_face, target_size)
    gray_face = gray_face.astype('float32')
    gray_face = gray_face / 255.0
    gray_face = gray_face - 0.5
    gray_face = gray_face * 2.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    return gray_face


def analyze_image(image_path):
    """Analyze an image and return emotion predictions."""
    import cv2
    import numpy as np
    from keras.models import load_model
    from PIL import Image
    
    # Load models
    emotion_classifier = load_model(str(MODEL_PATH), compile=False)
    face_detection = cv2.CascadeClassifier(str(DETECTION_MODEL_PATH))
    
    # Load and process image
    img = Image.open(image_path).convert('RGB')
    image = np.array(img).astype(np.float32) / 255.0
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_detection.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    face_detected = len(faces) > 0
    
    if face_detected:
        # Use largest face
        face_areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(face_areas)
        x, y, w, h = faces[largest_idx]
        gray_face = gray_image[y:y+h, x:x+w]
        face_location = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
    else:
        gray_face = gray_image
        h, w = gray_image.shape[:2]
        face_location = {"x": 0, "y": 0, "width": int(w), "height": int(h)}
    
    # Get model input size
    target_size = emotion_classifier.input_shape[1:3]
    processed_face = preprocess_face(gray_face, target_size)
    
    # Run inference
    emotion_predictions = emotion_classifier.predict(processed_face, verbose=0)[0]
    
    # Build detailed emotions dict for 6 basic emotions
    detailed_emotions = {emotion: 0.0 for emotion in BASIC_EMOTIONS}
    
    # Map model output to 6 basic emotions
    for idx, prob in enumerate(emotion_predictions):
        label = FER2013_LABELS[idx]
        if label in detailed_emotions:
            detailed_emotions[label] += float(prob)
    
    # Normalize to ensure probabilities sum to 1
    total_prob = sum(detailed_emotions.values())
    if total_prob > 0:
        detailed_emotions = {k: v / total_prob for k, v in detailed_emotions.items()}
    
    # Get dominant emotion
    dominant_idx = np.argmax(emotion_predictions)
    dominant_emotion = FER2013_LABELS[dominant_idx]
    dominant_confidence = float(emotion_predictions[dominant_idx])
    
    # Calculate uncertainty
    uncertainty = min(0.25, 0.05 + 0.2 * (1.0 - dominant_confidence))
    
    # Generate notes
    face_note = "" if face_detected else " (no face bounding box detected, analyzed full image)"
    if dominant_confidence > 0.5:
        notes = f"Strong {dominant_emotion} expression detected ({dominant_confidence:.1%} confidence){face_note}."
    elif dominant_confidence > 0.3:
        notes = f"Moderate {dominant_emotion} expression detected ({dominant_confidence:.1%} confidence){face_note}."
    else:
        notes = f"Weak facial expression. {dominant_emotion} is most likely ({dominant_confidence:.1%}){face_note}."
    
    return {
        "Happiness": float(round(detailed_emotions.get('Happiness', 0.0), 4)),
        "Sadness": float(round(detailed_emotions.get('Sadness', 0.0), 4)),
        "Fear": float(round(detailed_emotions.get('Fear', 0.0), 4)),
        "Anger": float(round(detailed_emotions.get('Anger', 0.0), 4)),
        "Disgust": float(round(detailed_emotions.get('Disgust', 0.0), 4)),
        "Surprise": float(round(detailed_emotions.get('Surprise', 0.0), 4)),
        "uncertainty_margin": float(round(uncertainty, 4)),
        "detailed_emotions": detailed_emotions,
        "dominant_emotion": dominant_emotion,
        "dominant_confidence": float(round(dominant_confidence, 4)),
        "detected_facial_cues": [dominant_emotion],
        "cue_confidence": float(round(dominant_confidence, 4)),
        "notes": notes,
        "disclaimer": "Emotion estimated using fer2013_mini_XCEPTION CNN model from latest-model. Results are probabilistic.",
        "model": "fer2013_mini_XCEPTION (latest-model)",
        "face_detected": face_detected,
        "face_location": face_location,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: keras_inference.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    try:
        result = analyze_image(image_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
