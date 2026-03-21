import logging
import os
import uuid
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from flask import current_app, jsonify, request
from flask_login import login_required
from werkzeug.utils import secure_filename

from . import bp
from dreamsApp.core.extra.clustering import cluster_keywords_for_all_users
from dreamsApp.core.extra.location_extractor import enrich_location
from dreamsApp.core.vector_store import vector_store
from dreamsApp.core.database import db_manager

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-V2")

_enrichment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="loc-enrich")

def _enrich_location_background(post_id, lat, lon):
    try:
        enrichment = enrich_location(lat, lon, model=model)
        if enrichment and "location_embedding" in enrichment:
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
    try:
        result = vector_store.store_keywords(user_id, post_id, keywords_with_vectors)
        if result:
            logger.info("Keywords stored in ChromaDB for post %s", post_id)
        else:
            logger.error("Failed to store keywords in ChromaDB for post %s", post_id)
    except Exception:
        logger.exception("Background keyword storage failed for post %s", post_id)

@bp.route('/upload', methods=['POST'])
@login_required
def upload_post():
    user_id = request.form.get('user_id')
    caption = request.form.get('caption')
    timestamp_str = request.form.get('timestamp', datetime.now().isoformat())
    image = request.files.get('image')

    missing = [k for k, v in {'caption': caption, 'image': image,'user_id': user_id}.items() if not v]
    if missing:
         return jsonify({'error': f"Missing required fields: {', '.join(missing)}"}), 400
    
    filename = secure_filename(image.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    upload_path = current_app.config['UPLOAD_FOLDER']
    image_path = os.path.join(upload_path, unique_filename)
    image.save(image_path)

    pipeline = current_app.dreams_pipeline
    pipeline_result = pipeline.process_new_post(user_id, image_path, caption, timestamp_str)
    
    post_doc = pipeline_result["post_doc"]
    keyword_type = pipeline_result.get("keyword_type")
    keywords_for_db = pipeline_result.get("keywords_for_db")
    keywords_with_vectors = pipeline_result.get("keywords_with_vectors")
    gps_data = pipeline_result.get("gps_data")

    # Handle keywords via SQLite arrays
    if keywords_for_db and keyword_type:
        col = f"{keyword_type}_json"
        with sqlite3.connect(db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            row = cursor.execute("SELECT * FROM keywords WHERE user_id = ?", (user_id,)).fetchone()
            new_kw = []
            if row and row[col]:
                new_kw = json.loads(row[col])
            
            new_kw.extend(keywords_for_db)
            
            if row:
                cursor.execute(f"UPDATE keywords SET {col} = ? WHERE user_id = ?", (json.dumps(new_kw), user_id))
            else:
                cursor.execute(f"INSERT INTO keywords (user_id, {col}) VALUES (?, ?)", (user_id, json.dumps(new_kw)))
                
            conn.commit()

    # Insert post into SQLite
    timestamp_dt = datetime.fromisoformat(post_doc['timestamp']) if isinstance(post_doc['timestamp'], str) else post_doc['timestamp']
    inserted_id = db_manager.insert_post(
        user_id=post_doc['user_id'],
        image_path=post_doc['image_path'],
        caption=post_doc['caption'],
        timestamp=timestamp_dt,
        sentiment_label=post_doc['sentiment']['label'],
        sentiment_score=post_doc['sentiment']['score']
    )

    if inserted_id == -1:
        return jsonify({'error': 'Failed to create post'}), 500

    if gps_data:
        _enrichment_executor.submit(
            _enrich_location_background,
            inserted_id,
            gps_data["lat"],
            gps_data["lon"]
        )

    if keywords_with_vectors:
        _enrichment_executor.submit(
            _store_keywords_background,
            user_id,
            str(inserted_id),
            keywords_with_vectors
        )

    return jsonify({
        'message': 'Post created successfully',
        'post_id': str(inserted_id),
        'user_id': user_id,
        'caption': caption,
        'timestamp': timestamp_dt.isoformat(),
        'image_path': image_path,
        'sentiment': post_doc['sentiment'],
        'generated_caption': post_doc.get('generated_caption', ''),
    }), 201

@bp.route("/run_clustering")
def manual_cluster():
    # Deprecated for SQLite context unless clustering logic is rewritten, leaving as a stub
    return "Clustering API endpoint is disabled during SQLite migration pending rewrite.", 403