from flask import request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from flask import current_app
from .  import bp

from app.utils.sentiment import get_image_caption_and_sentiment
from dreamsApp.app.utils.keywords import extract_keywords

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
    # keyword generation from the caption
    keywords = extract_keywords(caption)
    

    post_doc = {
        'user_id': user_id,
        'caption': caption,
        'timestamp': datetime.fromisoformat(timestamp),
        'image_path': image_path,
        'generated_caption': generated_caption,
        'sentiment' : sentiment,
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