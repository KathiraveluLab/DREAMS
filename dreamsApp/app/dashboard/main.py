from flask import render_template, request, url_for
from flask import current_app
from . import bp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
import threading
from flask_login import login_required, current_user
from wordcloud import WordCloud
from dreamsApp.core.extra.llms import generate
from flask import jsonify
import datetime
import json
import sqlite3
from dreamsApp.core.database import db_manager

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
    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()
        rows = cursor.execute("SELECT DISTINCT user_id FROM posts").fetchall()
        unique_users = [r[0] for r in rows if r[0]]
    return render_template('dashboard/main.html', users=unique_users)

@bp.route('/user/<string:target>', methods =['GET'])
@login_required
def profile(target):
    target_user_id = str(target)
    
    with sqlite3.connect(db_manager.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        user_posts_rows = cursor.execute("SELECT * FROM posts WHERE user_id = ? ORDER BY timestamp ASC", (target_user_id,)).fetchall()
        user_posts = [dict(row) for row in user_posts_rows]

    for post in user_posts:
        post['sentiment'] = {
            'label': post.get('sentiment_label'),
            'score': post.get('sentiment_score')
        }
        
        if isinstance(post['timestamp'], str):
            try:
                post['timestamp'] = datetime.datetime.fromisoformat(post['timestamp'])
            except ValueError:
                pass
                
        if 'chime_analysis_json' in post and post['chime_analysis_json']:
            try:
                post['chime_analysis'] = json.loads(post['chime_analysis_json'])
            except:
                post['chime_analysis'] = {}

    df = pd.DataFrame(user_posts)
    if df.empty:
        return render_template(
            'dashboard/profile.html', 
            plot_url=None, 
            chime_plot_url=None, 
            positive_wordcloud_url=None, 
            negative_wordcloud_url=None, 
            thematics={},
            user_id=target_user_id,
            latest_post=None
        )

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
    
    # Fetch keywords and thematics from SQLite
    with sqlite3.connect(db_manager.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        keywords_row = cursor.execute("SELECT * FROM keywords WHERE user_id = ?", (target_user_id,)).fetchone()
        thematics_row = cursor.execute("SELECT * FROM thematic_analysis WHERE user_id = ?", (target_user_id,)).fetchone()

    positive_keywords = []
    negative_keywords = []
    if keywords_row:
        pos_str = keywords_row['positive_keywords_json']
        neg_str = keywords_row['negative_keywords_json']
        pos_list = json.loads(pos_str) if pos_str else []
        neg_list = json.loads(neg_str) if neg_str else []
        positive_keywords = [item['keyword'] for item in pos_list]
        negative_keywords = [item['keyword'] for item in neg_list]
        
    if not thematics_row or not thematics_row['data_json']:
        try:
            thematics = generate(str(target_user_id), positive_keywords, negative_keywords)
        except Exception as e:
            current_app.logger.error(f"Error generating thematics: {e}")
            thematics = {}
    else:
        thematics = json.loads(thematics_row["data_json"])

    # Generate word clouds using helper function
    wordcloud_positive_data = generate_wordcloud_b64(positive_keywords, 'GnBu')
    wordcloud_negative_data = generate_wordcloud_b64(negative_keywords, 'OrRd')

    # Sort posts to get the latest one
    # The user_posts list is already sorted by timestamp ascending. The latest post is the last one.
    latest_post = user_posts[-1] if user_posts else None

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

@bp.route('/user/<string:target>/narrative', methods=['GET'])
@login_required
def narrative(target):
    """Render the Narrative Structure Analysis visualization page."""
    return render_template('dashboard/narrative.html', user_id=target)

@bp.route('/user/<string:target>/cluster_analysis', methods=['GET'])
@login_required
def cluster_analysis(target):
    """Render the Cluster Analysis visualization page."""
    return render_template('dashboard/cluster_analysis.html', user_id=target)

@bp.route('/clusters/<user_id>')
@login_required
def show_clusters(user_id):
    with sqlite3.connect(db_manager.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        user_doc = cursor.execute("SELECT * FROM keywords WHERE user_id = ?", (user_id,)).fetchone()

    if not user_doc or not user_doc['clustered_keywords_json']:
        return "No clusters found.", 404

    clustered_data = json.loads(user_doc['clustered_keywords_json'])

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
        with sqlite3.connect(db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            keywords_row = cursor.execute("SELECT * FROM keywords WHERE user_id = ?", (user_id,)).fetchone()

        positive_keywords = []
        negative_keywords = []
        if keywords_row:
            pos_str = keywords_row['positive_keywords_json']
            neg_str = keywords_row['negative_keywords_json']
            pos_list = json.loads(pos_str) if pos_str else []
            neg_list = json.loads(neg_str) if neg_str else []
            positive_keywords = [item['keyword'] for item in pos_list]
            negative_keywords = [item['keyword'] for item in neg_list]
    
        try:
            thematic_data = generate(str(user_id), positive_keywords, negative_keywords)
            current_app.logger.info("Refreshed thematic data:")
            
            return jsonify({
                'message': 'Thematics refreshed successfully',
                'thematic_data': thematic_data
            }), 200
        except Exception as e:
            current_app.logger.error(f"Error regenerating thematics: {e}")
            return jsonify({'error': str(e)}), 500
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
    
    if corrected_label not in VALID_CHIME_LABELS:
        return jsonify({'success': False, 'error': 'Invalid label value'}), 400
    
    now = datetime.datetime.utcnow().isoformat()
    
    try:
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Simple rate limiting logic placeholder (SQLite logic)
            # Just do the update
            
            result = cursor.execute(
                "UPDATE posts SET corrected_label = ? WHERE id = ? AND user_id = ?",
                (corrected_label, post_id, current_user.get_id())
            )
            conn.commit()
            
            if result.rowcount > 0:
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'error': 'Post not found or no change'}), 404
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500