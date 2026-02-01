from flask import render_template, request, url_for
from flask import current_app
from . import bp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
import threading
from flask_login import login_required, current_user
from wordcloud import WordCloud
from ..utils.llms import generate
from flask import jsonify
import datetime
from bson.objectid import ObjectId
from bson.errors import InvalidId

# Security: Whitelist of valid CHIME labels
VALID_CHIME_LABELS = {'Connectedness', 'Hope', 'Identity', 'Meaning', 'Empowerment', 'None'}

# Security: Rate limiting configuration
MAX_CORRECTIONS_PER_HOUR = 10

def generate_wordcloud_b64(keywords, colormap):
    """Refactor: Helper to generate base64 encoded word cloud image."""
    if not keywords:
        return None
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='#121212', 
        colormap=colormap
    ).generate(' '.join(keywords))
    
    buf = io.BytesIO()
    wordcloud.to_image().save(buf, 'png')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return data

@bp.route('/', methods =['GET'])
@login_required
def main():
    mongo = current_app.mongo['posts']
    unique_users = mongo.distinct('user_id')
    return render_template('dashboard/main.html', users=unique_users)

@bp.route('/user/<string:target>', methods =['GET'])
@login_required
def profile(target):
    mongo = current_app.mongo['posts']
    

    target_user_id = target
    user_posts = list(
    mongo.find({'user_id': target_user_id}).sort('timestamp', 1)
    )

    for post in user_posts:
        post['sentiment_label'] = post['sentiment']['label']
        post['sentiment_score'] = post['sentiment']['score']

    df = pd.DataFrame(user_posts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }
    df['score'] = df['sentiment_label'].map(sentiment_map)

    df = df.sort_values("timestamp")
    df["cumulative_score"] = df["score"].cumsum()
    df["rolling_avg"] = df["score"].rolling(window=5, min_periods=1).mean()
    df["ema_score"] = df["score"].ewm(span=5, adjust=False).mean()

    # Create user-friendly visual
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')

    plt.plot(df["timestamp"], df["cumulative_score"],
            label="Overall Emotional Journey", color="#90caf9", marker="o", alpha=0.5)

    plt.plot(df["timestamp"], df["rolling_avg"],
            label="5-Day Emotional Smoothing", color="#ffcc80", linestyle="--", marker="x")

    plt.plot(df["timestamp"], df["ema_score"],
            label="Recent Emotional Trend", color="#a5d6a7", linestyle="-", marker="s")

    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)

    #  Friendly and interpretive title and axis labels
    plt.title("How This Person’s Feelings Shifted Over Time", fontsize=14, color='white', fontweight='bold')
    plt.xlabel("When Posts Were Made", fontsize=12, color='#e0e0e0')
    plt.ylabel("Mood Score (Higher = Happier)", fontsize=12, color='#e0e0e0')

    #  Improve legend
    plt.legend(title="What the Lines Mean", fontsize=10, facecolor='#222', edgecolor='#444')
    plt.grid(color='#333333', linestyle=':', alpha=0.5)
    plt.xticks(rotation=45, color='#888888')
    plt.yticks(color='#888888')
    plt.tight_layout()

    #  Save to base64 for embedding
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#121212')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.clf() # Clear timeline plot

    # --- CHIME Radar Chart ---
    chime_counts = {
        "Connectedness": 0, "Hope": 0, "Identity": 0, 
        "Meaning": 0, "Empowerment": 0
    }
    
    # Optimize lookup for case-insensitivity
    chime_lookup = {k.lower(): k for k in chime_counts}

    for post in user_posts:
        # Prioritize user correction if available
        label_to_use = post.get('corrected_label')
        if not label_to_use and post.get('chime_analysis'):
            label_to_use = post['chime_analysis'].get('label', '')
            
        if label_to_use:
            original_key = chime_lookup.get(label_to_use.lower())
            if original_key:
                chime_counts[original_key] += 1
    
    categories = list(chime_counts.keys())
    values = list(chime_counts.values())
    
    # Radar chart requires closing the loop
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    # Setup the plot with dark theme colors to match dashboard
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(7, 7), facecolor='#121212') # Deep dark background
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor('#1e1e1e') # Slightly lighter plot area
    
    # Set radial limits based on data but with a minimum for visual clarity
    max_val = max(values) if any(values) else 1
    limit = max(2, max_val + 1)
    ax.set_ylim(0, limit)
    
    # Draw axes and labels
    plt.xticks(angles[:-1], categories, color='#00d4ff', size=12, fontweight='bold')
    ax.tick_params(colors='#888888') # Radial scale label color
    ax.grid(color='#444444', linestyle='--')

    # Plot data with vibrant blue fill and markers
    ax.plot(angles, values, color='#00d4ff', linewidth=3, linestyle='solid', marker='o', markersize=8)
    ax.fill(angles, values, color='#00d4ff', alpha=0.3)
    
    plt.title("Personal Recovery Footprint", size=18, color='white', pad=20, fontweight='bold')
    
    buf = io.BytesIO()
    # Save with specific facecolor to ensure transparency/consistency
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#121212')
    buf.seek(0)
    chime_plot_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.clf() # Clean up radar plot
    plt.style.use('default') # Reset style for next plots
    
    # Fetch keywords from MongoDB
    keywords_data = current_app.mongo['keywords'].find_one({'user_id': target_user_id})
    positive_keywords = [item['keyword'] for item in keywords_data.get('positive_keywords', [])] if keywords_data else []
    
    negative_keywords =  [item['keyword'] for item in keywords_data.get('negative_keywords', [])] if keywords_data else []
    
    # Check if thematics exist in the database
    thematics_data = current_app.mongo['thematic_analysis'].find_one({'user_id': str(target_user_id)})
    
    if not thematics_data or "data" not in thematics_data:
        thematics = generate(str(target_user_id), positive_keywords, negative_keywords)
    else:
        thematics = thematics_data["data"]

    # Generate word clouds using helper function
    wordcloud_positive_data = generate_wordcloud_b64(positive_keywords, 'GnBu')
    wordcloud_negative_data = generate_wordcloud_b64(negative_keywords, 'OrRd')

    # Sort posts to get the latest one
    user_posts.sort(key=lambda x: x['timestamp'], reverse=True)
    latest_post = user_posts[0] if user_posts else None

    return render_template(
        'dashboard/profile.html', 
        plot_url=plot_data, 
        chime_plot_url=chime_plot_data, 
        positive_wordcloud_url=wordcloud_positive_data, 
        negative_wordcloud_url=wordcloud_negative_data, 
        thematics=thematics,
        user_id=str(target_user_id),
        latest_post=latest_post  # Pass only the latest post for feedback
    )

@bp.route('/clusters/<user_id>')
@login_required
def show_clusters(user_id):
    mongo = current_app.mongo
    user_doc = mongo['keywords'].find_one({'user_id': user_id})

    if not user_doc or 'clustered_keywords' not in user_doc:
        return "No clusters found.", 404

    clustered_data = user_doc['clustered_keywords']

    # Group by sentiment → cluster → keywords
    grouped = {}
    for item in clustered_data:
        sentiment = item['sentiment']
        cluster = item['cluster']
        keyword = item['keyword']

        if sentiment not in grouped:
            grouped[sentiment] = {}
        if cluster not in grouped[sentiment]:
            grouped[sentiment][cluster] = []

        grouped[sentiment][cluster].append(keyword)

    return render_template('dashboard/clusters.html', grouped=grouped)

@bp.route('/refresh_thematic/<user_id>', methods=['POST'])
@login_required
def thematic_refresh(user_id):
    try:
        keywords_data = current_app.mongo['keywords'].find_one({'user_id': str(user_id)})
        positive_keywords = [item['keyword'] for item in keywords_data.get('positive_keywords', [])] if keywords_data else []

        negative_keywords = [item['keyword'] for item in keywords_data.get('negative_keywords', [])] if keywords_data else []
    
        thematic_data = generate(str(user_id), positive_keywords, negative_keywords)
        print("Refresed thematic data:")
        
        return jsonify({
            "success": True,
            "message": "Thematic updated successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@bp.route('/correct_chime', methods=['POST'])
@login_required
def correct_chime():
    data = request.get_json()
    post_id = data.get('post_id')
    corrected_label = data.get('corrected_label')
    
    if not all([post_id, corrected_label]):
        return jsonify({'success': False, 'error': 'Missing fields'}), 400
    
    # SECURITY: Validate ObjectId format
    try:
        post_object_id = ObjectId(post_id)
    except (InvalidId, TypeError):
        return jsonify({'success': False, 'error': 'Invalid post ID format'}), 400
    
    # SECURITY: Validate label is in allowed set
    if corrected_label not in VALID_CHIME_LABELS:
        return jsonify({'success': False, 'error': 'Invalid label value'}), 400
    
    mongo = current_app.mongo['posts']
    
    # SECURITY: Rate limiting - max corrections per user per hour
    one_hour_ago = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    recent_corrections = mongo.count_documents({
        'user_id': current_user.get_id(),
        'correction_timestamp': {'$gte': one_hour_ago}
    })
    
    if recent_corrections >= MAX_CORRECTIONS_PER_HOUR:
        return jsonify({'success': False, 'error': 'Rate limit exceeded. Try again later.'}), 429
    
    # Step 1: ALWAYS save the correction to the queue first
    now = datetime.datetime.utcnow()
    result = mongo.update_one(
        {'_id': post_object_id, 'user_id': current_user.get_id()},
        {
            '$set': {
                'corrected_label': corrected_label,  # Current correction
                'is_fl_processed': False,  # Added to queue
                'correction_timestamp': now
            },
            '$push': {
                # AUDIT TRAIL: Keep history of all corrections for auditing
                'correction_history': {
                    'label': corrected_label,
                    'timestamp': now,
                    'user_id': current_user.get_id()
                }
            }
        }
    )
    
    if result.modified_count > 0:
        # Step 2: Check if we should trigger training (non-blocking)
        _maybe_trigger_fl_training(current_app._get_current_object())
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Post not found or no change'}), 404


def _maybe_trigger_fl_training(app):
    """
    Check queue size and trigger training if threshold is met.
    Uses atomic database lock to ensure only ONE training runs at a time.
    If lock is busy, the correction is already saved - it will be processed next round.
    """
    FL_BATCH_SIZE = app.config.get('FL_BATCH_SIZE', 50)
    LOCK_TIMEOUT_HOURS = 2  # If lock is older than this, assume it's stale
    
    with app.app_context():
        mongo = app.mongo
        
        # Quick count check
        pending_count = mongo['posts'].count_documents({
            'corrected_label': {'$exists': True},
            'is_fl_processed': False
        })
        
        if pending_count < FL_BATCH_SIZE:
            return  # Not enough corrections yet, exit quickly
        
        # Try to acquire atomic lock
        # Only ONE request can successfully flip is_running from False to True
        lock_collection = mongo['fl_training_lock']
        
        # Ensure lock document exists (first-time setup)
        lock_collection.update_one(
            {'_id': 'singleton'},
            {'$setOnInsert': {'is_running': False}},
            upsert=True
        )
        
        # SECURITY: Check for stale lock (stuck for more than LOCK_TIMEOUT_HOURS)
        stale_threshold = datetime.datetime.now() - datetime.timedelta(hours=LOCK_TIMEOUT_HOURS)
        lock_collection.update_one(
            {'_id': 'singleton', 'is_running': True, 'started_at': {'$lt': stale_threshold}},
            {'$set': {'is_running': False, 'stale_reset_at': datetime.datetime.now()}}
        )
        
        # Atomically try to acquire lock
        lock_result = lock_collection.find_one_and_update(
            {'_id': 'singleton', 'is_running': False},
            {'$set': {'is_running': True, 'started_at': datetime.datetime.now()}},
            return_document=False  # Return the OLD document
        )
        
        if lock_result is None or lock_result.get('is_running', True):
            # Lock is busy - another training is running
            # Our correction is already saved in queue, it will be processed next round
            return
        
        # We got the lock! Start training in background thread
        def run_training_with_lock():
            # Wrap entire function in app_context since this runs in a separate thread
            with app.app_context():
                try:
                    # Import here to avoid circular dependency (fl_worker imports create_app)
                    from dreamsApp.app.fl_worker import run_federated_round
                    run_federated_round()
                except Exception as e:
                    # Log the error since daemon threads fail silently
                    import logging
                    logging.error(f"FL Training failed in background thread: {str(e)}", exc_info=True)
                finally:
                    # Always release lock when done (success or failure)
                    mongo['fl_training_lock'].update_one(
                        {'_id': 'singleton'},
                        {'$set': {'is_running': False, 'finished_at': datetime.datetime.now()}}
                    )
        
        thread = threading.Thread(target=run_training_with_lock, daemon=True)
        thread.start()