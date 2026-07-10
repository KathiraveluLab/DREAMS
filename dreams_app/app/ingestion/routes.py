import logging
import os
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from flask import current_app, jsonify, request
from PIL import Image, UnidentifiedImageError
from flask_login import login_required
from werkzeug.utils import secure_filename

from . import bp
from dreams_app.core.extra.clustering import cluster_keywords_for_all_users
from dreams_app.core.extra.location_extractor import enrich_location, extract_gps_from_image
from dreams_app.core.sentiment import get_chime_category
from dreams_app.core.extra.keywords import extract_keywords_and_vectors
from dreams_app.core.extra.image_captioning import get_image_caption
from dreams_app.core.vector_store import vector_store

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional heavy dependency
    SentenceTransformer = None

logger = logging.getLogger(__name__)
_location_model = None
_location_model_lock = threading.Lock()
MAX_IMAGE_PIXELS = 100_000_000


def _get_location_model():
    global _location_model
    if SentenceTransformer is None:
        return None
    with _location_model_lock:
        if _location_model is None:
            try:
                _location_model = SentenceTransformer("all-MiniLM-L6-V2")
            except Exception as e:
                logger.warning("Failed to load location embedding model: %s", e)
                _location_model = None
    return _location_model

# Background thread pool for non-blocking location enrichment.
# A single worker ensures Nominatim rate-limiting is respected naturally.
_enrichment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="loc-enrich")


def _enrich_location_background(post_id, lat, lon, mongo_uri, db_name):
    """Run reverse-geocoding + embedding in a background thread and
    update the MongoDB post document with the enrichment results.

    Runs outside the Flask request context, so it receives the raw
    connection parameters instead of ``current_app.mongo``.
    """
    try:
        from pymongo import MongoClient
        with MongoClient(mongo_uri) as client:
            db = client[db_name]

            enrichment = enrich_location(lat, lon, model=_get_location_model())
            if enrichment:
                # Strip the heavy semantic embedding before updating MongoDB
                mongo_enrichment = dict(enrichment)
                mongo_enrichment.pop("location_embedding", None)

                db["posts"].update_one(
                    {"_id": post_id},
                    {"$set": {f"location.{k}": v for k, v in mongo_enrichment.items()}},
                )

                # Push semantic location embedding to ChromaDB Multiplex Layer 2
                if "location_embedding" in enrichment:
                    vector_store.store_vector(
                        collection_name="layer_2_semantic",
                        doc_id=str(post_id),
                        embedding=enrichment["location_embedding"],
                        metadata={"location_text": enrichment.get("location_text", "")}
                    )

                logger.info("Location enrichment complete for post %s", post_id)
    except Exception:
        logger.exception("Background location enrichment failed for post %s", post_id)


def _store_embeddings_background(post_id, user_id, caption_embedding, image_embedding):
    """Push document embeddings to ChromaDB in background thread."""
    try:
        if caption_embedding:
            vector_store.store_vector(
                collection_name="text_embeddings",
                doc_id=str(post_id),
                embedding=caption_embedding,
                metadata={"user_id": user_id, "type": "caption"}
            )
        if image_embedding:
            vector_store.store_vector(
                collection_name="image_embeddings",
                doc_id=str(post_id),
                embedding=image_embedding,
                metadata={"user_id": user_id, "type": "image"}
            )
        logger.info("Embeddings stored in ChromaDB for post %s", post_id)

    except Exception:
        logger.exception("Background embeddings storage failed for post %s", post_id)


@bp.route('/upload', methods=['POST'])
@login_required
def upload_post():
    user_id = request.form.get('user_id')
    caption = request.form.get('caption')
    timestamp = request.form.get('timestamp', datetime.now(timezone.utc).isoformat())
    image = request.files.get('image')

    missing = [k for k, v in {'caption': caption, 'image': image, 'user_id': user_id}.items() if not v]
    if missing:
         return jsonify({'error': f"Missing required fields: {', '.join(missing)}"}), 400

    filename = secure_filename(image.filename)
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg', 'gif'})
    if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in allowed:
        return jsonify({'error': 'Unsupported file extension'}), 400

    try:
        # Security: Validate image from request stream before any disk write.
        with Image.open(image.stream) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed safety limits")
            # Fully decode pixel data to ensure image is valid and not corrupt.
            img.load()

        # Rewind request stream after validation and save validated content.
        image.stream.seek(0)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        upload_path = current_app.config['UPLOAD_FOLDER']
        image_path = os.path.join(upload_path, unique_filename)
        image.save(image_path)
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        image_path = locals().get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'error': 'Uploaded file is not a valid image'}), 400

    # Delegate the heavy AI extraction sequence to the global pipeline.
    # Validation above ensures we only process decodable image files.
    pipeline = current_app.dreams_pipeline
    pipeline_result = pipeline.process_new_post(user_id, image_path, caption, timestamp)

    post_doc = pipeline_result["post_doc"]
    caption_embedding = pipeline_result.get("caption_embedding")
    image_embedding = pipeline_result.get("image_embedding")
    sentiment = post_doc.get("sentiment", {})

    # 1. Image Captioning
    try:
        generated_caption = get_image_caption(image_path)
    except Exception as e:
        logger.error(f"Image captioning failed: {e}")
        generated_caption = caption

    # 2. CHIME Analysis
    text_for_analysis = caption if (caption and caption.strip()) else generated_caption
    try:
        chime_result = get_chime_category(text_for_analysis)
    except Exception as e:
        logger.error(f"CHIME analysis failed: {e}")
        chime_result = None

    # 3. Keywords Extraction
    keywords_with_vectors = []
    keyword_type = None
    keywords_for_mongo = []

    if sentiment.get('label') in ('positive', 'negative'):
        try:
            keywords_with_vectors = extract_keywords_and_vectors(generated_caption)
            keyword_type = f"{sentiment['label']}_keywords"
            if keywords_with_vectors:
                keywords_for_mongo = [
                    {k: v for k, v in kw.items() if k != "embedding"}
                    for kw in keywords_with_vectors
                ]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")

    # 4. Location Extraction locally
    gps_data = extract_gps_from_image(image_path)

    # Add newly computed fields into the post_doc
    if gps_data:
        post_doc['location'] = gps_data
    post_doc['generated_caption'] = generated_caption
    post_doc['chime_analysis'] = chime_result

    mongo = current_app.mongo

    # We maintain previous keyword updating logic so older flask dashboards don't break
    if keywords_for_mongo and keyword_type:
        kw_update_result = mongo['keywords'].update_one(
            {'user_id': user_id},
            {'$push': {keyword_type: {'$each': keywords_for_mongo}}},
            upsert=True
        )

        if kw_update_result.upserted_id:
            if keyword_type == 'negative_keywords':
                mongo['keywords'].update_one(
                    {'_id': kw_update_result.upserted_id},
                    {'$set': {'positive_keywords': []}}
                )
            elif keyword_type == 'positive_keywords':
                mongo['keywords'].update_one(
                    {'_id': kw_update_result.upserted_id},
                    {'$set': {'negative_keywords': []}}
                )

    insert_result = mongo['posts'].insert_one(post_doc)

    if not insert_result.acknowledged:
        return jsonify({'error': 'Failed to create post'}), 500

    # Fire-and-forget: enrich location in a background thread so the
    # response is not blocked by Nominatim rate-limiting (~1.1 s).
    if gps_data:
        mongo_uri = current_app.config.get("MONGO_URI", "mongodb://localhost:27017")
        db_name = mongo.name
        _enrichment_executor.submit(
            _enrich_location_background,
            insert_result.inserted_id,
            gps_data["lat"],
            gps_data["lon"],
            mongo_uri,
            db_name,
        )

    # Fire-and-forget: push document embeddings into ChromaDB
    if caption_embedding or image_embedding:
        _enrichment_executor.submit(
            _store_embeddings_background,
            str(insert_result.inserted_id),
            user_id,
            caption_embedding,
            image_embedding
        )

    return jsonify({
        'message': 'Post created successfully',
        'post_id': str(insert_result.inserted_id),
        'user_id': user_id,
        'caption': caption,
        'timestamp': post_doc['timestamp'].isoformat(),
        'image_path': image_path,
        'sentiment': post_doc['sentiment'],
        'generated_caption': post_doc['generated_caption'],
    }), 201


@bp.route("/run_clustering")
def manual_cluster():
    cluster_keywords_for_all_users(current_app.mongo['keywords'])
    return "Clustering done"
