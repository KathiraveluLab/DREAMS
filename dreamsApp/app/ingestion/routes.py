import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from flask import current_app, jsonify, request
from werkzeug.utils import secure_filename

from . import bp
from ..utils.keywords import extract_keywords_and_vectors
from ..utils.clustering import cluster_keywords_for_all_users
from ..utils.location_extractor import extract_gps_from_image, enrich_location
from ..utils.sentiment import (
    get_chime_category,
    get_image_caption_and_sentiment,
    select_text_for_analysis,
)
from ..utils.vector_store import vector_store

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-V2")

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

            enrichment = enrich_location(lat, lon, model=model)
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


def _store_keywords_background(user_id, post_id, keywords_with_vectors):
    """Push keywords to ChromaDB in background thread with error handling."""
    try:
        result = vector_store.store_keywords(user_id, post_id, keywords_with_vectors)
        if result:
            logger.info("Keywords stored in ChromaDB for post %s", post_id)
        else:
            logger.error("Failed to store keywords in ChromaDB for post %s (store_keywords returned False)", post_id)
    except Exception:
        logger.exception("Background keyword storage failed for post %s", post_id)


@bp.route('/upload', methods=['POST'])
def upload_post():
    user_id = request.form.get('user_id')
    caption = request.form.get('caption')
    timestamp = request.form.get('timestamp', datetime.now().isoformat())
    image = request.files.get('image')

    if not all([user_id, caption, timestamp, image]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    filename = secure_filename(image.filename)
    upload_path = current_app.config['UPLOAD_FOLDER']
    image_path = os.path.join(upload_path, filename)
    image.save(image_path)

    # Extract GPS from EXIF if available
    gps_data = extract_gps_from_image(image_path)
    
    analysis_result = get_image_caption_and_sentiment(image_path, caption)
    
    sentiment = analysis_result["sentiment"]
    generated_caption = analysis_result["imgcaption"]

    # Refactor: Use shared selection logic to determine which text to analyze for recovery
    text_for_analysis = select_text_for_analysis(caption, generated_caption)
    chime_result = get_chime_category(text_for_analysis)
    
    # keyword generation from the caption
    
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
        # Strip out the heavy 384-dimensional vector arrays before saving to MongoDB
        keywords_for_mongo = [
            {k: v for k, v in kw.items() if k != "embedding"}
            for kw in keywords_with_vectors
        ]

        mongo = current_app.mongo
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
                
    # We will defer pushing keywords to ChromaDB until we have the post_id


    post_doc = {
        'user_id': user_id,
        'caption': caption,
        'timestamp': datetime.fromisoformat(timestamp),
        'image_path': image_path,
        'generated_caption': generated_caption,
        'sentiment' : sentiment,
        'chime_analysis': chime_result,
        'location': gps_data,
    }

    mongo = current_app.mongo
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

    # Fire-and-forget: push extracted keywords into ChromaDB
    if keywords_with_vectors:
        _enrichment_executor.submit(
            _store_keywords_background,
            user_id,
            str(insert_result.inserted_id),
            keywords_with_vectors
        )

    return jsonify({
        'message': 'Post created successfully',
        'post_id': str(insert_result.inserted_id),
        'user_id': user_id,
        'caption': caption,
        'timestamp': datetime.fromisoformat(timestamp),
        'image_path': image_path,
        'sentiment': sentiment,
        'generated_caption': generated_caption,
    }), 201

    
@bp.route("/run_clustering")
def manual_cluster():
    cluster_keywords_for_all_users()
    return "Clustering done"