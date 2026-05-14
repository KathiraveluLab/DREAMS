from flask import request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import logging
from flask import current_app
from .  import bp

logger = logging.getLogger(__name__)


from ..utils.sentiment import get_image_caption_and_sentiment, get_chime_category, select_text_for_analysis
from ..utils.keywords import extract_keywords_and_vectors
from ..utils.clustering import cluster_keywords_for_all_users
from ..utils.places365_classifier import classify_scene

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-V2")


# --- Default fallback constants (Pipeline stability improvement) ---
_DEFAULT_SENTIMENT = {"label": "neutral", "score": 0.0}
_DEFAULT_SCENE = {"scene_type": "unknown", "scene_confidence": 0.0, "scene_raw_top3": []}


@bp.route('/upload', methods=['POST'])
def upload_post():
    user_id = request.form.get('user_id')
    timestamp = request.form.get('timestamp', datetime.now().isoformat())
    image = request.files.get('image')

    # Pipeline stability improvement: caption defaults to empty string if missing
    caption = request.form.get('caption', '').strip()
    if not caption:
        logger.warning("Missing or empty caption for upload by user_id=%s — defaulting to empty string", user_id)

    if not all([user_id, timestamp, image]):
        return jsonify({'error': 'Missing required fields (user_id, timestamp, image)'}), 400
    
    filename = secure_filename(image.filename)
    upload_path = current_app.config['UPLOAD_FOLDER']
    image_path = os.path.join(upload_path, filename)
    image.save(image_path)

    # Pipeline stability improvement: emotion model failure handling
    try:
        result = get_image_caption_and_sentiment(image_path, caption)
        sentiment = result["sentiment"]
        generated_caption = result["imgcaption"]
    except Exception as e:
        logger.error("Emotion/caption model failed for %s: %s — using neutral defaults", image_path, e)
        sentiment = _DEFAULT_SENTIMENT.copy()
        generated_caption = ""

    # Refactor: Use shared selection logic to determine which text to analyze for recovery
    text_for_analysis = select_text_for_analysis(caption, generated_caption)

    # Pipeline stability improvement: CHIME analysis failure handling
    try:
        chime_result = get_chime_category(text_for_analysis)
    except Exception as e:
        logger.error("CHIME analysis failed: %s — using uncategorized default", e)
        chime_result = {"label": "Uncategorized", "score": 0.0}
    
    # keyword generation from the caption
    # Pipeline stability improvement: keyword extraction failure handling
    try:
        # Extract keyword + vector pairs
        if sentiment['label'] == 'negative':
            keywords_with_vectors = extract_keywords_and_vectors(generated_caption)
            keyword_type = 'negative_keywords'
        elif sentiment['label'] == 'positive':
            keywords_with_vectors = extract_keywords_and_vectors(generated_caption)
            keyword_type = 'positive_keywords'
        else:
            keywords_with_vectors = []
            keyword_type = None

        if keywords_with_vectors:
            mongo = current_app.mongo
            kw_result = mongo['keywords'].update_one(
                {'user_id': user_id},
                {'$push': {keyword_type: {'$each': keywords_with_vectors}}},
                upsert=True
            )

            # Pipeline stability improvement: fixed stale variable bug — now uses kw_result
            if kw_result.upserted_id:
                if keyword_type == 'negative_keywords':
                    mongo['keywords'].update_one(
                        {'_id': kw_result.upserted_id},
                        {'$set': {'positive_keywords': []}}
                    )
                elif keyword_type == 'positive_keywords':
                    mongo['keywords'].update_one(
                        {'_id': kw_result.upserted_id},
                        {'$set': {'negative_keywords': []}}
                    )
    except Exception as e:
        logger.error("Keyword extraction/storage failed for user_id=%s: %s — skipping keywords", user_id, e)

    # Scene classification — Basic integration
    # Fine-tuning on Alaska imagery and CLIP secondary verification planned for GSoC coding period
    # Pipeline stability improvement: scene classification fallback
    try:
        scene_result = classify_scene(image_path)
        scene_type = scene_result.get("scene_type", "unknown")
        scene_confidence = scene_result.get("scene_confidence", 0.0)
        scene_raw_top3 = scene_result.get("scene_raw_top3", [])

        # Pipeline stability improvement: low confidence threshold fallback
        if scene_confidence < 0.4 and scene_type != "unknown":
            logger.warning(
                "Scene classification confidence %.4f below threshold for %s — falling back to 'unknown'",
                scene_confidence, image_path
            )
            scene_type = "unknown"
    except Exception as e:
        logger.warning("Scene classification failed for %s: %s — using unknown fallback", image_path, e)
        scene_type = "unknown"
        scene_confidence = 0.0
        scene_raw_top3 = []

    post_doc = {
        'user_id': user_id,
        'caption': caption,
        'timestamp': datetime.fromisoformat(timestamp),
        'image_path': image_path,
        'generated_caption': generated_caption,
        'sentiment' : sentiment,
        'chime_analysis': chime_result,
        'scene_type': scene_type,
        'scene_confidence': scene_confidence,
        'scene_raw_top3': scene_raw_top3,
    }

    mongo = current_app.mongo
    result = mongo['posts'].insert_one(post_doc)

    if result.acknowledged:
        return jsonify({'message': 'Post created successfully',
                        'post_id': str(result.inserted_id),
                        'user_id': user_id,
                        'caption': caption,
                        'timestamp': datetime.fromisoformat(timestamp),
                        'image_path': image_path,
                        'sentiment' : sentiment,
                        'generated_caption': generated_caption
                        }), 201
    else:
        return jsonify({'error': 'Failed to create post'}), 500


    # Pipeline stability improvement: non-blocking DB insertion
    try:
        mongo = current_app.mongo
        result = mongo['posts'].insert_one(post_doc)

        if result.acknowledged:
            return jsonify({'message': 'Post created successfully',
                            'post_id': str(result.inserted_id),
                            'user_id': user_id,
                            'caption': caption,
                            'timestamp': datetime.fromisoformat(timestamp),
                            'image_path': image_path,
                            'sentiment' : sentiment,
                            'generated_caption': generated_caption
                            }), 201
        else:
            return jsonify({'error': 'Failed to create post'}), 500
    except Exception as e:
        logger.error("Database insertion failed for user_id=%s: %s", user_id, e)
        return jsonify({'error': 'Failed to create post', 'details': str(e)}), 500
    
@bp.route("/run_clustering")
def manual_cluster():
    cluster_keywords_for_all_users()
    return "Clustering done"