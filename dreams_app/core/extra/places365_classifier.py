# Basic integration — fine-tuning on Alaska imagery and CLIP secondary verification planned for GSoC coding period

"""
Places365 Scene Classification for DREAMS

Classifies images into recovery-relevant scene categories using the
Places365 ResNet50 pretrained model. Scene types are mapped from
365 raw place categories into 6 DREAMS-specific categories.
"""

import os
import hashlib
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Module-level cache for model and labels
_model = None
_labels = None
MODEL_SHA256_ENV = "PLACES365_MODEL_SHA256"
LABELS_SHA256 = "2affba635eb657e7ca95f4e6cc69bd9fac29ef4c32aeb83cafdfcd06ec6a1ea6"

# The 6 DREAMS scene categories
VALID_SCENE_TYPES = {
    "clinical_or_institutional",
    "faith_community",
    "recovery_support",
    "residential_or_transitional",
    "shelter_or_dropin",
    "outdoor_or_wilderness",
}

# Pipeline stability improvement: default fallback result for scene classification failures
_DEFAULT_SCENE_RESULT = {
    "scene_type": "unknown",
    "scene_confidence": 0.0,
    "scene_raw_top3": [],
}

# Mapping from raw Places365 labels to DREAMS categories.
# Labels not present here will not contribute to any category.
CATEGORY_MAPPING = {
    # clinical_or_institutional
    "hospital": "clinical_or_institutional",
    "hospital_room": "clinical_or_institutional",
    "clinic": "clinical_or_institutional",
    "pharmacy": "clinical_or_institutional",
    "dentists_office": "clinical_or_institutional",
    "operating_room": "clinical_or_institutional",
    "waiting_room": "clinical_or_institutional",
    "laboratory": "clinical_or_institutional",
    "nursing_home": "clinical_or_institutional",

    # faith_community
    "church/indoor": "faith_community",
    "church/outdoor": "faith_community",
    "mosque/indoor": "faith_community",
    "mosque/outdoor": "faith_community",
    "temple/east_asia": "faith_community",
    "temple/south_asia": "faith_community",
    "synagogue/indoor": "faith_community",
    "chapel": "faith_community",
    "cathedral/indoor": "faith_community",
    "cathedral/outdoor": "faith_community",
    "abbey": "faith_community",

    # recovery_support
    "community_center": "recovery_support",
    "recreation_room": "recovery_support",
    "conference_room": "recovery_support",
    "meeting_room": "recovery_support",
    "classroom": "recovery_support",
    "lecture_room": "recovery_support",
    "office": "recovery_support",
    "gym/indoor": "recovery_support",

    # residential_or_transitional
    "bedroom": "residential_or_transitional",
    "living_room": "residential_or_transitional",
    "house": "residential_or_transitional",
    "apartment_building/outdoor": "residential_or_transitional",
    "kitchen": "residential_or_transitional",
    "dining_room": "residential_or_transitional",
    "bathroom": "residential_or_transitional",
    "home_office": "residential_or_transitional",
    "porch": "residential_or_transitional",
    "yard": "residential_or_transitional",
    "balcony/interior": "residential_or_transitional",
    "balcony/exterior": "residential_or_transitional",
    "garage/indoor": "residential_or_transitional",
    "residential_neighborhood": "residential_or_transitional",

    # shelter_or_dropin
    "tent/outdoor": "shelter_or_dropin",
    "dormitory": "shelter_or_dropin",
    "youth_hostel": "shelter_or_dropin",
    "motel": "shelter_or_dropin",
    "locker_room": "shelter_or_dropin",
    "campsite": "shelter_or_dropin",

    # outdoor_or_wilderness
    "park": "outdoor_or_wilderness",
    "forest_path": "outdoor_or_wilderness",
    "forest/broadleaf": "outdoor_or_wilderness",
    "forest/needleleaf": "outdoor_or_wilderness",
    "mountain": "outdoor_or_wilderness",
    "mountain_path": "outdoor_or_wilderness",
    "mountain_snowy": "outdoor_or_wilderness",
    "field/cultivated": "outdoor_or_wilderness",
    "field/wild": "outdoor_or_wilderness",
    "lake/natural": "outdoor_or_wilderness",
    "river": "outdoor_or_wilderness",
    "ocean": "outdoor_or_wilderness",
    "beach": "outdoor_or_wilderness",
    "creek": "outdoor_or_wilderness",
    "sky": "outdoor_or_wilderness",
    "valley": "outdoor_or_wilderness",
    "snowfield": "outdoor_or_wilderness",
    "glacier": "outdoor_or_wilderness",
    "trail": "outdoor_or_wilderness",
    "swamp": "outdoor_or_wilderness",
    "waterfall": "outdoor_or_wilderness",
}


# Image preprocessing: resize to 256, center crop to 224, normalize with ImageNet stats
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path, expected_sha256, artifact_name):
    if not expected_sha256:
        raise RuntimeError(f"{artifact_name} SHA-256 checksum is required before download")

    actual_sha256 = _sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(f"{artifact_name} SHA-256 checksum verification failed")


def _download_verified(url, path, expected_sha256, artifact_name):
    torch.hub.download_url_to_file(url, path)
    try:
        _verify_sha256(path, expected_sha256, artifact_name)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise


def _load_model():
    """
    Load the Places365 ResNet50 model and category labels.

    Downloads the pretrained weights from the Places365 project and
    the category label file on first call, then caches them at module
    level for subsequent calls.

    Returns:
        tuple: (model, labels) where model is the ResNet50 with Places365
               weights and labels is a list of 365 scene category strings.

    Pipeline stability improvement: raises on failure so callers can
    fall back to the default unknown-scene result.
    """
    global _model, _labels

    if _model is not None and _labels is not None:
        return _model, _labels

    logger.info("Loading Places365 ResNet50 model...")

    # Use the Places365 pretrained ResNet50
    model_file = "resnet50_places365.pth.tar"
    model_url = "https://places2.csail.mit.edu/models_places365/" + model_file

    # Download model weights if not cached
    cache_dir = os.path.join(torch.hub.get_dir(), "places365")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, model_file)

    if not os.path.exists(model_path):
        logger.info("Downloading Places365 weights...")
        _download_verified(
            model_url,
            model_path,
            os.environ.get(MODEL_SHA256_ENV),
            "Places365 model weights",
        )
    elif os.environ.get(MODEL_SHA256_ENV):
        _verify_sha256(
            model_path,
            os.environ[MODEL_SHA256_ENV],
            "Places365 model weights",
        )

    # Build ResNet50 architecture with 365 output classes
    from torchvision.models import resnet50
    model = resnet50(num_classes=365)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Download category labels
    labels_file = "categories_places365.txt"
    labels_url = "https://raw.githubusercontent.com/csailvision/places365/master/" + labels_file
    labels_path = os.path.join(cache_dir, labels_file)

    if not os.path.exists(labels_path):
        logger.info("Downloading Places365 labels...")
        _download_verified(labels_url, labels_path, LABELS_SHA256, "Places365 labels")
    else:
        _verify_sha256(labels_path, LABELS_SHA256, "Places365 labels")

    labels = []
    with open(labels_path, "r") as f:
        for line in f:
            # Format: /a/airfield 0  or  /c/church/indoor 0
            # Split on "/" and rejoin everything after the prefix letter
            label = line.strip().split(" ")[0]
            label = "/".join(label.split("/")[2:])
            labels.append(label)

    _model = model
    _labels = labels
    logger.info("Places365 model loaded successfully.")
    return _model, _labels


def classify_scene(image_path):
    """
    Classify the scene type of an image using Places365 ResNet50.

    Takes an image file path, runs it through the pretrained Places365
    model, and maps the top prediction to one of 6 DREAMS recovery
    categories. If the model's confidence is below 0.4, returns
    "unknown" as the scene type.

    Args:
        image_path (str): Absolute or relative path to the image file.

    Returns:
        dict: A dictionary with the following keys:
            - scene_type (str): One of the 6 DREAMS categories or "unknown".
            - scene_confidence (float): Confidence score between 0 and 1.
            - scene_raw_top3 (list): Top 3 raw Places365 labels with
              their confidence scores, each as
              {"label": str, "confidence": float}.

    Pipeline stability improvement: returns default unknown-scene result
    on any failure (model loading, image preprocessing, inference).
    """
    # Pipeline stability improvement: model loading failure fallback
    try:
        model, labels = _load_model()
    except Exception as e:
        logger.warning("Scene classification model loading failed: %s — returning unknown fallback", e)
        return _DEFAULT_SCENE_RESULT.copy()

    # Pipeline stability improvement: image preprocessing failure fallback
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
    except Exception as e:
        logger.warning("Scene classification image preprocessing failed for %s: %s — returning unknown fallback", image_path, e)
        return _DEFAULT_SCENE_RESULT.copy()

    # Pipeline stability improvement: inference failure fallback
    try:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    except Exception as e:
        logger.warning("Scene classification inference failed for %s: %s — returning unknown fallback", image_path, e)
        return _DEFAULT_SCENE_RESULT.copy()

    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    scene_raw_top3 = []
    for i in range(3):
        scene_raw_top3.append({
            "label": labels[top3_idx[i].item()],
            "confidence": round(float(top3_prob[i].item()), 4),
        })

    # Determine DREAMS scene category from top predictions
    # Walk through top predictions and use the first one that maps
    scene_type = "unknown"
    scene_confidence = round(float(top3_prob[0].item()), 4)

    # Check confidence threshold
    if scene_confidence < 0.4:
        # Pipeline stability improvement: log low-confidence fallback
        logger.warning(
            "Scene classification confidence %.4f below threshold for %s — returning unknown",
            scene_confidence, image_path
        )
        return {
            "scene_type": "unknown",
            "scene_confidence": scene_confidence,
            "scene_raw_top3": scene_raw_top3,
        }

    # Try to map top predictions to a DREAMS category
    for i in range(3):
        raw_label = labels[top3_idx[i].item()]
        if raw_label in CATEGORY_MAPPING:
            scene_type = CATEGORY_MAPPING[raw_label]
            scene_confidence = round(float(top3_prob[i].item()), 4)
            break

    return {
        "scene_type": scene_type,
        "scene_confidence": scene_confidence,
        "scene_raw_top3": scene_raw_top3,
    }
