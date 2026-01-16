from flask import request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from flask import current_app
from .  import bp


from ..utils.sentiment import get_image_caption_and_sentiment, get_chime_category, select_text_for_analysis
from ..utils.keywords import extract_keywords_and_vectors
from ..utils.clustering import cluster_keywords_for_all_users

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-V2")


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
    result = get_image_caption_and_sentiment(image_path, caption)
    
    sentiment = result["sentiment"]
    generated_caption = result["imgcaption"]

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
        mongo = current_app.mongo
        result = mongo['keywords'].update_one(
            {'user_id': user_id},
            {'$push': {keyword_type: {'$each': keywords_with_vectors}}},
            upsert=True
        )

    if result.upserted_id:
        if keyword_type == 'negative_keywords':
            mongo['keywords'].update_one(
                {'_id': result.upserted_id},
                {'$set': {'positive_keywords': []}}
            )
        elif keyword_type == 'positive_keywords':
            mongo['keywords'].update_one(
                {'_id': result.upserted_id},
                {'$set': {'negative_keywords': []}}
            )
    

    post_doc = {
        'user_id': user_id,
        'caption': caption,
        'timestamp': datetime.fromisoformat(timestamp),
        'image_path': image_path,
        'generated_caption': generated_caption,
        'sentiment' : sentiment,
        'chime_analysis': chime_result  # Store the new object
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
    
@bp.route("/run_clustering")
def manual_cluster():
    cluster_keywords_for_all_users()
    return "Clustering done"