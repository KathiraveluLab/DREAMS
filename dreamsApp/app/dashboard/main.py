from flask import render_template, request, url_for
from flask import current_app
from . import bp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
from flask_login import login_required
from wordcloud import WordCloud
from ..utils.llms import generate
from flask import jsonify

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
        if post.get('chime_analysis'):
            label = post['chime_analysis'].get('label', '').lower()
            original_key = chime_lookup.get(label)
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

    return render_template('dashboard/profile.html', plot_url=plot_data, chime_plot_url=chime_plot_data, positive_wordcloud_url=wordcloud_positive_data, negative_wordcloud_url=wordcloud_negative_data, thematics=thematics,user_id=str(target_user_id))

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