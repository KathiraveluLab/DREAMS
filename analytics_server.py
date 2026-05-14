#!/usr/bin/env python3

import json
import os
import tempfile
import logging
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, render_template_string, redirect, url_for, send_file
from flask import Flask, jsonify, render_template_string, redirect, url_for, make_response
import random

# Add dreamsApp to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dreamsApp.analytics import (
    EmotionEvent,
    EmotionTimeline,
    Episode,
    build_narrative_graph,
    segment_timeline_to_episodes,
    EmotionTimelineSerializer,
    EpisodeSerializer,
    TemporalNarrativeGraphSerializer,
    ContentAddressedStore,
    StructuralCache,
    build_frontend_payload,
    SCHEMA_VERSION,
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Initialize cache
cache_dir = Path(tempfile.gettempdir()) / "dreams_analytics_cache"
store = ContentAddressedStore(cache_dir)
cache = StructuralCache(store)

# Check for optional dependencies at startup (for perceptual emotion analysis)
_PERCEPTUAL_DEPS_AVAILABLE = True
_PERCEPTUAL_DEPS_ERROR = None
try:
    from PIL import Image
    import numpy as np
    from ml.latest_emotion_model import estimate_emotion_from_image, compare_image_estimates
except ImportError as e:
    _PERCEPTUAL_DEPS_AVAILABLE = False
    _PERCEPTUAL_DEPS_ERROR = "Perceptual emotion analysis dependencies are unavailable"
    logger.warning("Perceptual emotion analysis dependencies not available", exc_info=True)
    _PERCEPTUAL_DEPS_ERROR = str(e)
    print(f"WARNING: Perceptual emotion analysis dependencies not available: {e}")
    print("   The /api/perceptual-emotion and /api/compare-images endpoints will return fallback data.")

# Check for text sentiment analysis dependencies
_TEXT_SENTIMENT_AVAILABLE = True
_TEXT_SENTIMENT_ERROR = None
try:
    from ml.text_sentiment import analyze_text_sentiment
except ImportError as e:
    _TEXT_SENTIMENT_AVAILABLE = False
    _TEXT_SENTIMENT_ERROR = str(e)
    print(f"WARNING: Text sentiment analysis not available: {e}")

# Simulated users with emotion data
SAMPLE_USERS = {
    "user_001": "Alice Johnson",
    "user_002": "Bob Smith", 
    "user_003": "Carol Williams",
}


# Alice's special images
ALICE_IMAGES = [
    "/static/images/download.jpeg",
    "/static/images/download (1).jpeg",
    "/static/images/download (2).jpeg"
]

# Bob Smith's sample data (user_002) — loaded from bob-smith.json
BOB_SMITH_JSON_PATH = Path(__file__).parent / "sample-data" / "bob-smith" / "bob-smith.json"
BOB_SMITH_DATA = {}
BOB_SMITH_IMAGES = []
BOB_SMITH_CAPTIONS = {}  # image_path -> description
try:
    with open(BOB_SMITH_JSON_PATH) as f:
        BOB_SMITH_DATA = json.load(f)
    for photo in BOB_SMITH_DATA.get("photos", []):
        img_path = f"/static/sample-data/bob-smith/{photo['filename']}"
        BOB_SMITH_IMAGES.append(img_path)
        BOB_SMITH_CAPTIONS[img_path] = photo.get("description", "")
    print(f"✅ Loaded bob-smith sample data: {len(BOB_SMITH_IMAGES)} images")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Could not load or parse bob-smith.json: {e}")

def generate_sample_posts(user_id: str) -> list:
    """Generate sample posts with emotions for a user."""
    random.seed(hash(user_id))  # Deterministic per user
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    emotions = ["positive", "neutral", "negative"]
    posts = []
    
    # Generate 20-40 posts over 30 days
    num_posts = random.randint(20, 40)
    
    for i in range(num_posts):
        day_offset = random.randint(0, 29)
        hour_offset = random.randint(0, 23)
        minute_offset = random.randint(0, 59)
        
        timestamp = base_time + timedelta(days=day_offset, hours=hour_offset, minutes=minute_offset)
        
        # Bias towards positive for some users, negative for others
        if "001" in user_id:
            weights = [0.6, 0.3, 0.1]  # More positive
        elif "002" in user_id:
            weights = [0.2, 0.5, 0.3]  # More neutral
        else:
            weights = [0.3, 0.3, 0.4]  # More negative
        
        emotion = random.choices(emotions, weights=weights)[0]
        
        post = {
            '_id': f"post_{user_id}_{i}",
            'user_id': user_id,
            'timestamp': timestamp,
            'caption': f"Sample caption {i} for {user_id}",
            'sentiment': {
                'label': emotion,
                'score': random.uniform(0.5, 1.0)
            }
        }
        
        posts.append(post)
    
    # Sort posts chronologically
    posts = sorted(posts, key=lambda p: p['timestamp'])
    
    # Add images to Alice's first 3 chronological posts
    if user_id == "user_001":
        for i, image_path in enumerate(ALICE_IMAGES):
            if i < len(posts):
                posts[i]['image'] = image_path

    # Add images and captions from bob-smith.json to Bob Smith's posts.
    # Insert dedicated posts at the very start of the timeline, each spaced
    # >6 hours apart so they form Episode 1, Episode 2, Episode 3.
    if user_id == "user_002" and BOB_SMITH_IMAGES:
        earliest = posts[0]['timestamp'] if posts else base_time
        # Place image posts before all other posts, each 1 day apart
        for i, image_path in enumerate(BOB_SMITH_IMAGES):
            img_time = earliest - timedelta(days=len(BOB_SMITH_IMAGES) - i)
            category = BOB_SMITH_DATA['photos'][i].get("category", "neutral")
            sentiment_label = {"happy": "positive", "sad": "negative"}.get(category, "neutral")
            img_post = {
                '_id': f"post_{user_id}_img_{i}",
                'user_id': user_id,
                'timestamp': img_time,
                'caption': BOB_SMITH_CAPTIONS.get(image_path, ""),
                'image': image_path,
                'sentiment': {
                    'label': sentiment_label,
                    'score': 0.9
                }
            }
            posts.insert(0, img_post)
        # Re-sort to keep chronological order
        posts = sorted(posts, key=lambda p: p['timestamp'])

    return posts


def build_timeline_from_posts(user_posts: list, user_id: str) -> EmotionTimeline:
    """Convert posts to EmotionTimeline.

    Pipeline stability improvement: handles posts with missing or
    malformed sentiment dicts by defaulting to neutral / 0.0.
    """
    events = []
    for post in user_posts:
        # Pipeline stability improvement: default sentiment for missing data
        sentiment = post.get('sentiment') or {}
        events.append(EmotionEvent(
            timestamp=post['timestamp'],
            emotion_label=sentiment.get('label', 'neutral'),
            score=sentiment.get('score', 0.0),
            source_id=str(post.get('_id', '')),
        ))
    
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    return EmotionTimeline(subject_id=user_id, events=tuple(events_sorted))


# ============== HTML TEMPLATES ==============

MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DREAMS Analytics Demo</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #121212; color: #e0e0e0; }
        .card { background: #1e1e1e; border: 1px solid #333; }
        .card:hover { transform: translateY(-4px); box-shadow: 0 0.5rem 1rem rgba(255,255,255,0.05); }
        .btn-narrative { background: linear-gradient(135deg, #4a90d9 0%, #357abd 100%); border: none; }
    </style>
</head>
<body>
    <div class="container py-5">
        <h1 class="text-center mb-4">DREAMS Analytics Demo</h1>
        
        <div class="row justify-content-center">
            {% for user_id, name in users.items() %}
            <div class="col-md-4 mb-4">
                <div class="card p-4 text-center">
                    <h5 class="text-white">{{ name }}</h5>
                    <p class="text-white small">{{ user_id }}</p>
                    <div class="d-grid gap-2">
                        <a href="{{ url_for('narrative_view', user_id=user_id) }}" class="btn btn-narrative text-white">View Narrative</a>
                        <a href="{{ url_for('api_timeline', user_id=user_id) }}" class="btn btn-outline-secondary btn-sm">API: Timeline</a>
                        <a href="{{ url_for('api_frontend_payload', user_id=user_id) }}" class="btn btn-outline-secondary btn-sm">API: Frontend Payload</a>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="text-center mt-5">
            <h4>API Endpoints</h4>
            <ul class="list-unstyled">
                <li><code>GET /api/timeline/&lt;user_id&gt;</code> - Timeline with fingerprint</li>
                <li><code>GET /api/episodes/&lt;user_id&gt;</code> - Episodes with stable IDs</li>
                <li><code>GET /api/narrative-graph/&lt;user_id&gt;</code> - Full graph</li>
                <li><code>GET /api/frontend-payload/&lt;user_id&gt;</code> - Frontend-ready payload</li>
                <li><code>GET /api/cache-status/&lt;user_id&gt;</code> - Cache status</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

NARRATIVE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Narrative - {{ user_id }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body { background: #121212; color: #e0e0e0; }
        .section { padding: 2rem 0; border-top: 1px solid #333; }
        .stat-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; }
        .stat-value { font-size: 2rem; font-weight: bold; color: #4a90d9; }
        .stat-label { font-size: 0.9rem; color: #888; }
        .episode-node { background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border: 2px solid #4a90d9; border-radius: 12px; padding: 1rem; margin: 0.5rem; min-width: 180px; text-align: center; transition: all 0.3s ease; }
        .episode-node:hover { transform: scale(1.05); box-shadow: 0 0 20px rgba(74, 144, 217, 0.4); }
        .episode-node img { cursor: pointer; }
        .episode-node img:hover { opacity: 0.8; }
        .episode-id { font-family: monospace; font-size: 0.7rem; color: #888; margin-top: 0.5rem; }
        .graph-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; padding: 2rem; background: #0a0a0a; border-radius: 1rem; }
        .edge-info { background: #2a2a2a; padding: 0.5rem 1rem; border-radius: 0.5rem; margin: 0.25rem; display: inline-block; }
        .edge-info.adjacent { border-left: 4px solid #4ad974; }
        .edge-info.overlapping { border-left: 4px solid #d9a74a; }
        .fingerprint { font-family: monospace; font-size: 0.8rem; color: #4a90d9; background: #1a1a2e; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
        .btn-back { background: #333; border: none; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.85); overflow-y: auto; }
        .modal.show { display: flex; align-items: flex-start; justify-content: center; padding: 2rem 1rem; }
        .modal-content { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 2px solid #4a90d9; border-radius: 1rem; padding: 1.5rem; max-width: 480px; width: 95%; position: relative; margin: auto; }
        .modal-close { position: absolute; top: 0.75rem; right: 0.75rem; font-size: 1.5rem; color: #e0e0e0; cursor: pointer; background: none; border: none; z-index: 10; }
        .modal-close:hover { color: #4a90d9; }
        .modal-episode-title { font-size: 1.1rem; font-weight: bold; color: #4a90d9; text-align: center; margin-bottom: 0.75rem; }
        .modal-image-wrap { text-align: center; margin-bottom: 0.75rem; }
        .modal-image-wrap img { max-width: 100%; max-height: 220px; border-radius: 0.75rem; object-fit: cover; }
        .modal-caption { font-style: italic; color: #ccc; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 0.6rem 0.75rem; border-radius: 0.5rem; border-left: 3px solid #4a90d9; margin-bottom: 0.75rem; display: none; line-height: 1.4; }
        .prob-section-title { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
        .prob-row { display: flex; align-items: center; margin: 0.35rem 0; }
        .prob-label { width: 70px; font-weight: bold; font-size: 0.85rem; }
        .prob-bar-container { flex: 1; margin: 0 0.5rem; }
        .prob-bar { height: 20px; border-radius: 10px; transition: width 0.5s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 0.4rem; font-size: 0.75rem; color: white; min-width: 2rem; }
        .prob-positive { background: linear-gradient(90deg, #2e7d32 0%, #4ad974 100%); }
        .prob-neutral { background: linear-gradient(90deg, #357abd 0%, #4a90d9 100%); }
        .prob-negative { background: linear-gradient(90deg, #bd3535 0%, #d94a4a 100%); }
        .perceptual-badge { display: inline-block; background: linear-gradient(135deg, #ff6b6b 0%, #ffa502 100%); color: #000; padding: 0.15rem 0.6rem; border-radius: 1rem; font-size: 0.7rem; font-weight: bold; }
        .disclaimer { font-size: 0.7rem; color: #666; font-style: italic; margin-top: 0.5rem; padding: 0.4rem; background: rgba(255,255,255,0.03); border-radius: 0.4rem; }
        .caption-text { font-style: italic; color: #aaa; margin: 0.25rem 0; font-size: 0.8rem; max-width: 180px; word-wrap: break-word; }
        /* Loading spinner */
        .loading-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 0.75rem; z-index: 5; }
        .loading-overlay.hidden { display: none; }
        .loading-spinner { width: 40px; height: 40px; border: 4px solid rgba(74, 144, 217, 0.3); border-top: 4px solid #4a90d9; border-radius: 50%; animation: spin 1s linear infinite; }
        .loading-text { color: #4a90d9; font-size: 0.85rem; margin-top: 0.75rem; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .modal-image-wrap { position: relative; }
        /* View Graph button */
        .btn-view-graph { background: linear-gradient(135deg, #6c5ce7 0%, #a55eea 100%); border: none; color: #fff; padding: 0.35rem 1rem; border-radius: 0.6rem; font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; display: none; vertical-align: middle; }
        .btn-view-graph:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(108,92,231,0.4); color: #fff; }
        /* Graph modal */
        .graph-modal { display: none; position: fixed; z-index: 1100; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); overflow-y: auto; }
        .graph-modal.show { display: flex; align-items: flex-start; justify-content: center; padding: 2rem 1rem; }
        .graph-modal-content { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 2px solid #6c5ce7; border-radius: 1rem; padding: 2rem; max-width: 720px; width: 95%; position: relative; margin: auto; }
        .graph-modal-close { position: absolute; top: 0.75rem; right: 0.75rem; font-size: 1.5rem; color: #e0e0e0; cursor: pointer; background: none; border: none; z-index: 10; }
        .graph-modal-close:hover { color: #6c5ce7; }
        .graph-modal-title { font-size: 1.3rem; font-weight: bold; color: #6c5ce7; text-align: center; margin-bottom: 0.25rem; }
        .graph-modal-subtitle { font-size: 0.85rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
        .graph-chart-wrap { background: #0a0a1a; border-radius: 0.75rem; padding: 1.25rem; }
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2>Temporal Narrative: {{ user_id }}</h2>
            <a href="/" class="btn btn-back text-white">← Back</a>
        </div>

        <div class="row mb-4" id="stats-container">
            <div class="col-md-3"><div class="stat-card"><div class="stat-value" id="stat-events">-</div><div class="stat-label">Events</div></div></div>
            <div class="col-md-3"><div class="stat-card"><div class="stat-value" id="stat-episodes">-</div><div class="stat-label">Episodes</div></div></div>
            <div class="col-md-3"><div class="stat-card"><div class="stat-value" id="stat-edges">-</div><div class="stat-label">Connections</div></div></div>
            <div class="col-md-3"><div class="stat-card"><div class="stat-value" id="stat-cached">-</div><div class="stat-label">Cached</div></div></div>
        </div>
        <div class="text-center mb-4"><span class="fingerprint" id="graph-fingerprint">Loading...</span></div>

        <div class="section">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h4 class="mb-0">Episode Network <small class="text-muted">(click images for perceptual analysis)</small></h4>
                <button class="btn btn-view-graph" id="btn-view-graph" onclick="openGraphModal()">📋 View Client Report</button>
            </div>
            <div class="graph-container" id="graph-container"><div class="text-muted">Loading...</div></div>
        </div>

        <div class="section">
            <h4>Connections</h4>
            <div id="edges-container" class="text-center"><span class="text-muted">Loading...</span></div>
        </div>

    </div>

    <!-- Episode Emotion Distribution Modal -->
    <div id="graphModal" class="graph-modal">
        <div class="graph-modal-content">
            <button class="graph-modal-close" onclick="closeGraphModal()">&times;</button>
            <div class="graph-modal-title">Episode Emotion Distribution</div>
            <div class="graph-modal-subtitle">Emotion percentages per episode for {{ user_name }}</div>
            <div class="graph-chart-wrap">
                <canvas id="emotionBarChart"></canvas>
            </div>
        </div>
    </div>
    
    <div id="emotionModal" class="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            
            <!-- 1. User Name & Episode -->
            <div id="modal-user" style="text-align:center; font-size:0.85rem; color:#888; margin-bottom:0.2rem;"></div>
            <div class="modal-episode-title" id="modal-title">Episode 1</div>
            
            <!-- 2. Image -->
            <div class="modal-image-wrap">
                <img id="modal-image" src="" alt="Analyzed image">
                <div id="image-loading-overlay" class="loading-overlay hidden">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Analyzing emotion...</div>
                </div>
            </div>
            
            <!-- 3. Description / Caption -->
            <div id="modal-caption" class="modal-caption"></div>
            
            <!-- 4. Emotion Percentages -->
            <div class="prob-section-title">Emotion Analysis <span class="perceptual-badge" id="modal-badge">🔬 PERCEPTUAL</span></div>
            <div id="prob-display">
                <div class="prob-row">
                    <span class="prob-label" style="color: #4ad974;">Happy</span>
                    <div class="prob-bar-container">
                        <div id="prob-bar-positive" class="prob-bar prob-positive" style="width: 0%;">0%</div>
                    </div>
                </div>
                <div class="prob-row">
                    <span class="prob-label" style="color: #4a90d9;">Neutral</span>
                    <div class="prob-bar-container">
                        <div id="prob-bar-neutral" class="prob-bar prob-neutral" style="width: 0%;">0%</div>
                    </div>
                </div>
                <div class="prob-row">
                    <span class="prob-label" style="color: #d94a4a;">Sad</span>
                    <div class="prob-bar-container">
                        <div id="prob-bar-negative" class="prob-bar prob-negative" style="width: 0%;">0%</div>
                    </div>
                </div>
            </div>
            
            <div id="notes-text" class="small text-muted mt-2"></div>
            
            <div class="disclaimer">
                Note: This is a probabilistic estimate. No confidence is ever 100%.
            </div>
        </div>
    </div>

    <script>
        const userId = "{{ user_id }}";
        let allEpisodeData = [];
        let currentChart = null;
        
        async function loadData() {
            const timelineRes = await fetch(`/api/timeline/${userId}`, { cache: 'no-store' });
            const timeline = await timelineRes.json();
            
            document.getElementById('stat-events').textContent = timeline.event_count;
            document.getElementById('graph-fingerprint').textContent = `Fingerprint: ${timeline.fingerprint}`;

            const payloadRes = await fetch(`/api/frontend-payload/${userId}`, { cache: 'no-store' });
            let payload = await payloadRes.json();

            // Auto-retry once if a user expected to have images returns none on first load
            const expectedImageUser = userId === 'user_001' || userId === 'user_002';
            const hasAnyImages = payload.nodes && payload.nodes.some(node => node.images && node.images.length > 0);
            if (expectedImageUser && !hasAnyImages) {
                await new Promise(resolve => setTimeout(resolve, 250));
                const retryRes = await fetch(`/api/frontend-payload/${userId}?_=${Date.now()}`, { cache: 'no-store' });
                payload = await retryRes.json();
            }
            
            document.getElementById('stat-episodes').textContent = payload.node_count;
            document.getElementById('stat-edges').textContent = payload.edge_count;
            document.getElementById('stat-cached').textContent = '✓';
            document.getElementById('graph-fingerprint').textContent = `Graph ID: ${payload.graph_id}`;
            
            allEpisodeData = payload.nodes.map(node => ({
                ...node,
                events: timeline.events.filter(e => {
                    const eventTime = new Date(e.timestamp);
                    const start = new Date(node.start_time_iso);
                    const end = new Date(node.end_time_iso);
                    return eventTime >= start && eventTime < end;
                })
            }));
            
            renderGraph(payload);
            renderEdges(payload);

            // Report is available for all users; data is rebuilt on modal open
            document.getElementById('btn-view-graph').style.display = 'inline-block';
        }
        
        function renderGraph(payload) {
            const container = document.getElementById('graph-container');
            container.innerHTML = '';
            if (!payload.nodes || !payload.nodes.length) {
                container.innerHTML = '<div class="text-muted">No episodes</div>';
                return;
            }
            payload.nodes.forEach((node, idx) => {
                const div = document.createElement('div');
                div.className = 'episode-node';
                
                let imageHTML = '';
                if (node.images && node.images.length > 0) {
                    // Display all images in the episode
                    const imagesContainer = node.images.map((imgSrc, imgIdx) => {
                        const imgPath = getRelativeImagePath(imgSrc);
                        const normalizedImgSrc = encodeURI(imgSrc);
                        return `<img src="${normalizedImgSrc}" alt="Episode ${node.index + 1} Image ${imgIdx + 1}" 
                            style="width: ${node.images.length > 1 ? '48%' : '100%'}; height: ${node.images.length > 1 ? '80px' : '120px'}; object-fit: cover; border-radius: 8px; cursor: pointer;" 
                            onclick="showPerceptualAnalysis('${imgPath}', '${imgSrc}', ${idx})">`;
                    }).join('');
                    imageHTML = `<div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 0.5rem;">${imagesContainer}</div>`;
                    
                    // Show captions from bob-smith.json if available
                    if (node.captions) {
                        node.images.forEach(imgSrc => {
                            const caption = node.captions[imgSrc];
                            if (caption) {
                                const truncated = caption.length > 70 ? caption.substring(0, 70) + '...' : caption;
                                imageHTML += `<div class="caption-text">"${truncated}"</div>`;
                            }
                        });
                    }
                }
                
                div.innerHTML = `
                    ${imageHTML}
                    <div><strong>Episode ${node.index + 1}</strong></div>
                    <div>${new Date(node.start_time_iso).toLocaleDateString()}</div>
                    <div class="small">${node.event_count} events</div>
                    <div class="small">${Math.round(node.duration_seconds / 60)} min</div>
                    <div class="episode-id">${node.id.substring(0, 8)}...</div>
                `;
                container.appendChild(div);
            });
        }
        
        function renderEdges(payload) {
            const container = document.getElementById('edges-container');
            container.innerHTML = '';
            if (!payload.edges || !payload.edges.length) {
                container.innerHTML = '<span class="text-muted">No connections</span>';
                return;
            }
            payload.edges.forEach(edge => {
                const div = document.createElement('div');
                div.className = `edge-info ${edge.relation}`;
                div.innerHTML = `Episode ${edge.source_index + 1} → Episode ${edge.target_index + 1}: <strong>${edge.relation}</strong>`;
                container.appendChild(div);
            });
        }
        
        let activeAnalysisToken = 0;
        let analysisTimeoutIds = [];

        function clearAnalysisTimeouts() {
            analysisTimeoutIds.forEach(id => clearTimeout(id));
            analysisTimeoutIds = [];
        }

        async function showPerceptualAnalysis(imagePath, imageSrc, episodeIndex) {
            const analysisToken = ++activeAnalysisToken;
            clearAnalysisTimeouts();

            document.getElementById('modal-image').src = imageSrc;
            document.getElementById('emotionModal').classList.add('show');
            
            // Reset displays
            document.getElementById('prob-bar-positive').style.width = '0%';
            document.getElementById('prob-bar-positive').textContent = '0%';
            document.getElementById('prob-bar-neutral').style.width = '0%';
            document.getElementById('prob-bar-neutral').textContent = '0%';
            document.getElementById('prob-bar-negative').style.width = '0%';
            document.getElementById('prob-bar-negative').textContent = '0%';
            document.getElementById('notes-text').textContent = '';
            
            // Show loading overlay
            const loadingOverlay = document.getElementById('image-loading-overlay');
            loadingOverlay.classList.remove('hidden');
            
            // Check if text-based emotion data is available (user_002 / Bob Smith)
            const episodeData = allEpisodeData[episodeIndex];
            let textEmotionData = null;
            let caption = null;
            
            if (episodeData && episodeData.text_emotions && episodeData.text_emotions[imageSrc]) {
                textEmotionData = episodeData.text_emotions[imageSrc];
            }
            if (episodeData && episodeData.captions && episodeData.captions[imageSrc]) {
                caption = episodeData.captions[imageSrc];
            }
            
            // 1. User name & Episode title
            const userName = userId in {"user_001":1,"user_002":1,"user_003":1} ? {"user_001":"Alice Johnson","user_002":"Bob Smith","user_003":"Carol Williams"}[userId] : userId;
            document.getElementById('modal-user').textContent = userName;
            document.getElementById('modal-title').textContent = `Episode ${episodeIndex + 1}`;
            
            // 3. Caption
            const captionEl = document.getElementById('modal-caption');
            if (caption) {
                captionEl.textContent = caption;
                captionEl.style.display = 'block';
            } else {
                captionEl.style.display = 'none';
            }
            
            // 4. Get emotion data (prefer perceptual model; fallback to text when needed)
            let data = null;
            let usedTextEmotion = false;
            try {
                const response = await fetch(`/api/perceptual-emotion/${encodeURIComponent(imagePath)}`, { cache: 'no-store' });
                const perceptualData = await response.json();
                if (analysisToken !== activeAnalysisToken) return;

                if (!perceptualData.error) {
                    data = perceptualData;
                } else if (textEmotionData) {
                    data = textEmotionData;
                    usedTextEmotion = true;
                } else {
                    loadingOverlay.classList.add('hidden');
                    document.getElementById('modal-badge').textContent = '🔬 PERCEPTUAL';
                    document.getElementById('notes-text').textContent = 'Error: ' + perceptualData.error;
                    return;
                }
            } catch (error) {
                if (analysisToken !== activeAnalysisToken) return;
                if (textEmotionData) {
                    data = textEmotionData;
                    usedTextEmotion = true;
                } else {
                    loadingOverlay.classList.add('hidden');
                    document.getElementById('modal-badge').textContent = '🔬 PERCEPTUAL';
                    document.getElementById('notes-text').textContent = 'Error: ' + error.message;
                    return;
                }
            }

            if (usedTextEmotion) {
                await new Promise(resolve => setTimeout(resolve, 300));
                if (analysisToken !== activeAnalysisToken) return;
                document.getElementById('modal-badge').textContent = '📝 TEXT SENTIMENT';
            } else {
                document.getElementById('modal-badge').textContent = '🔬 PERCEPTUAL';
            }
            loadingOverlay.classList.add('hidden');
            
            // Animate bars
            const t1 = setTimeout(() => {
                if (analysisToken !== activeAnalysisToken) return;
                document.getElementById('prob-bar-positive').style.width = `${data.positive * 100}%`;
                document.getElementById('prob-bar-positive').textContent = `${(data.positive * 100).toFixed(1)}%`;
            }, 100);
            const t2 = setTimeout(() => {
                if (analysisToken !== activeAnalysisToken) return;
                document.getElementById('prob-bar-neutral').style.width = `${data.neutral * 100}%`;
                document.getElementById('prob-bar-neutral').textContent = `${(data.neutral * 100).toFixed(1)}%`;
            }, 200);
            const t3 = setTimeout(() => {
                if (analysisToken !== activeAnalysisToken) return;
                document.getElementById('prob-bar-negative').style.width = `${data.negative * 100}%`;
                document.getElementById('prob-bar-negative').textContent = `${(data.negative * 100).toFixed(1)}%`;
            }, 300);
            analysisTimeoutIds.push(t1, t2, t3);
            
            document.getElementById('notes-text').textContent = data.notes || '';
        }
        
        function closeModal() {
            activeAnalysisToken++;
            clearAnalysisTimeouts();
            document.getElementById('emotionModal').classList.remove('show');
            document.getElementById('image-loading-overlay').classList.add('hidden');
        }

        // ---- Episode Emotion Distribution Graph ----
        let emotionGraphData = [];  // [{label, happy, neutral, sad}]
        let emotionBarChart = null;

        function getRelativeImagePath(imgSrc) {
            return decodeURIComponent(
                imgSrc.replace('/static/images/', '').replace('/static/sample-data/', '')
            );
        }

        async function buildEmotionGraphData() {
            const episodeRows = await Promise.all(allEpisodeData.map(async (ep, idx) => {
                const hasImages = Array.isArray(ep.images) && ep.images.length > 0;
                const hasDescriptions = hasImages && ep.images.some(imgSrc => {
                    const caption = ep.captions && ep.captions[imgSrc];
                    return typeof caption === 'string' && caption.trim().length > 0;
                });

                // If image or description is missing, show 0% for that episode
                if (!hasImages || !hasDescriptions) {
                    return {
                        label: `Episode ${idx + 1}`,
                        happy: 0,
                        neutral: 0,
                        sad: 0,
                    };
                }

                // Build episode percentages from perceptual model outputs only
                const estimates = await Promise.all(ep.images.map(async (imgSrc) => {
                    const relativePath = getRelativeImagePath(imgSrc);
                    try {
                        const response = await fetch(`/api/perceptual-emotion/${encodeURIComponent(relativePath)}`, { cache: 'no-store' });
                        const result = await response.json();
                        if (result.error) return null;
                        if (
                            typeof result.positive !== 'number' ||
                            typeof result.neutral !== 'number' ||
                            typeof result.negative !== 'number'
                        ) {
                            return null;
                        }
                        return result;
                    } catch (err) {
                        return null;
                    }
                }));

                const valid = estimates.filter(Boolean);
                if (!valid.length) {
                    return {
                        label: `Episode ${idx + 1}`,
                        happy: 0,
                        neutral: 0,
                        sad: 0,
                    };
                }

                const sums = valid.reduce(
                    (acc, item) => {
                        acc.positive += item.positive;
                        acc.neutral += item.neutral;
                        acc.negative += item.negative;
                        return acc;
                    },
                    { positive: 0, neutral: 0, negative: 0 }
                );

                return {
                    label: `Episode ${idx + 1}`,
                    happy: Math.round((sums.positive / valid.length) * 100),
                    neutral: Math.round((sums.neutral / valid.length) * 100),
                    sad: Math.round((sums.negative / valid.length) * 100),
                };
            }));

            emotionGraphData = episodeRows;
        }

        async function openGraphModal() {
            const graphButton = document.getElementById('btn-view-graph');
            const originalButtonText = graphButton.textContent;
            graphButton.disabled = true;
            graphButton.textContent = '⏳ Loading Report...';

            await buildEmotionGraphData();

            graphButton.disabled = false;
            graphButton.textContent = originalButtonText;
            document.getElementById('graphModal').classList.add('show');
            renderEmotionBarChart();
        }
        function closeGraphModal() {
            document.getElementById('graphModal').classList.remove('show');
            if (emotionBarChart) { emotionBarChart.destroy(); emotionBarChart = null; }
        }
        document.addEventListener('click', function(e) {
            if (e.target === document.getElementById('graphModal')) closeGraphModal();
        });

        function renderEmotionBarChart() {
            if (emotionBarChart) { emotionBarChart.destroy(); emotionBarChart = null; }
            const ctx = document.getElementById('emotionBarChart').getContext('2d');
            const labels = emotionGraphData.map(d => d.label);
            emotionBarChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Happy',
                            data: emotionGraphData.map(d => d.happy),
                            backgroundColor: 'rgba(74, 217, 116, 0.85)',
                            borderColor: '#4ad974',
                            borderWidth: 1,
                            borderRadius: 4,
                        },
                        {
                            label: 'Neutral',
                            data: emotionGraphData.map(d => d.neutral),
                            backgroundColor: 'rgba(74, 144, 217, 0.85)',
                            borderColor: '#4a90d9',
                            borderWidth: 1,
                            borderRadius: 4,
                        },
                        {
                            label: 'Sad',
                            data: emotionGraphData.map(d => d.sad),
                            backgroundColor: 'rgba(217, 74, 74, 0.85)',
                            borderColor: '#d94a4a',
                            borderWidth: 1,
                            borderRadius: 4,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    animation: { duration: 800, easing: 'easeOutQuart' },
                    plugins: {
                        legend: {
                            labels: { color: '#e0e0e0', font: { size: 13 }, padding: 20 },
                        },
                        tooltip: {
                            callbacks: {
                                label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%`
                            }
                        },
                    },
                    scales: {
                        x: {
                            ticks: { color: '#aaa', font: { size: 12, weight: 'bold' } },
                            grid: { color: 'rgba(255,255,255,0.06)' },
                        },
                        y: {
                            min: 0,
                            max: 100,
                            ticks: { color: '#aaa', stepSize: 20, callback: v => v + '%' },
                            grid: { color: 'rgba(255,255,255,0.08)' },
                            title: { display: true, text: 'Percentage', color: '#888' },
                        },
                    },
                },
            });
        }
        
        document.addEventListener('click', function(event) {
            if (event.target === document.getElementById('emotionModal')) {
                closeModal();
            }
        });
        
        document.addEventListener('DOMContentLoaded', loadData);
    </script>
</body>
</html>
"""


# ============== ROUTES ==============

@app.route('/')
def index():
    return render_template_string(MAIN_TEMPLATE, users=SAMPLE_USERS)


@app.route('/narrative/<user_id>')
def narrative_view(user_id: str):
    if user_id not in SAMPLE_USERS:
        return jsonify({'error': 'User not found'}), 404
    response = make_response(render_template_string(
        NARRATIVE_TEMPLATE,
        user_id=user_id,
        user_name=SAMPLE_USERS[user_id],
    ))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/timeline/<user_id>')
def api_timeline(user_id: str):
    posts = generate_sample_posts(user_id)
    timeline = build_timeline_from_posts(posts, user_id)
    
    payload = EmotionTimelineSerializer.serialize(timeline)
    cache.put(payload)
    
    return jsonify({
        'user_id': user_id,
        'schema_version': SCHEMA_VERSION,
        'fingerprint': timeline.fingerprint(),
        'cached': cache.is_valid(timeline.fingerprint()),
        'event_count': len(timeline),
        'duration_seconds': timeline.duration(),
        'temporal_bounds': {
            'start': timeline.events[0].timestamp.isoformat() if timeline.events else None,
            'end': timeline.events[-1].timestamp.isoformat() if timeline.events else None,
        },
        'events': [
            {'timestamp': e.timestamp.isoformat(), 'emotion_label': e.emotion_label, 'score': e.score}
            for e in timeline.events
        ]
    })


@app.route('/api/episodes/<user_id>')
def api_episodes(user_id: str):
    posts = generate_sample_posts(user_id)
    timeline = build_timeline_from_posts(posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(hours=6))
    
    return jsonify({
        'user_id': user_id,
        'schema_version': SCHEMA_VERSION,
        'episode_count': len(episodes),
        'episodes': [
            {
                'episode_id': ep.episode_id,
                'start_time': ep.start_time.isoformat(),
                'end_time': ep.end_time.isoformat(),
                'event_count': len(ep),
            }
            for ep in episodes
        ]
    })


@app.route('/api/narrative-graph/<user_id>')
def api_narrative_graph(user_id: str):
    posts = generate_sample_posts(user_id)
    timeline = build_timeline_from_posts(posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(hours=6))
    graph = build_narrative_graph(episodes, adjacency_threshold=timedelta(hours=12))
    
    payload = TemporalNarrativeGraphSerializer.serialize(graph)
    cache.put(payload)
    
    return jsonify({
        'user_id': user_id,
        'graph_id': graph.graph_id,
        'node_count': graph.node_count(),
        'edge_count': graph.edge_count(),
        'nodes': [{'index': i, 'episode_id': n.episode_id} for i, n in enumerate(graph.nodes)],
        'edges': [{'source': e.source_index, 'target': e.target_index, 'relation': e.relation.value} for e in graph.edges],
    })


@app.route('/static/images/<path:filename>')
def serve_image(filename: str):
    """
    Serve images from the images directory.
    
    WARNING: This uses Flask's send_from_directory which is suitable for development only.
    In production, use a dedicated static file server (Nginx, Apache) or CDN.
    Configure your reverse proxy to serve /static/images/ directly from the images directory.
    """
    images_dir = Path(__file__).parent / "images"
    safe_path = None
    try:
        safe_path = Path(os.path.realpath(images_dir / filename))
        safe_path.relative_to(images_dir.resolve())
    except (ValueError, OSError):
        return jsonify({'error': 'Invalid path'}), 400

    return send_file(safe_path)


@app.route('/static/sample-data/<path:filename>')
def serve_sample_data(filename: str):
    """Serve images from the sample-data directory."""
    from flask import send_from_directory
    sample_dir = Path(__file__).parent / "sample-data"
    try:
        safe_path = (sample_dir / filename).resolve()
        safe_path.relative_to(sample_dir.resolve())
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400
    return send_from_directory(sample_dir, filename)


@app.route('/api/frontend-payload/<user_id>')
def api_frontend_payload(user_id: str):
    posts = generate_sample_posts(user_id)
    timeline = build_timeline_from_posts(posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(hours=6))
    graph = build_narrative_graph(episodes, adjacency_threshold=timedelta(hours=12))
    frontend = build_frontend_payload(graph)
    
    result = frontend.to_dict()
    result['user_id'] = user_id
    
    # Add images to nodes by matching episode timestamps with posts
    for node_data in result['nodes']:
        # Find posts in this episode's time range
        start = datetime.fromisoformat(node_data['start_time_iso'])
        end = datetime.fromisoformat(node_data['end_time_iso'])
        episode_images = []
        for post in posts:
            if start <= post['timestamp'] < end and 'image' in post:
                episode_images.append(post['image'])
        if episode_images:
            node_data['images'] = episode_images

    # Add captions and text emotion analysis for all users
    # Pipeline stability improvement: per-node try/except so one bad node
    # does not crash the entire frontend payload
    for node_data in result['nodes']:
        if 'images' not in node_data:
            continue

        try:
            start = datetime.fromisoformat(node_data['start_time_iso'])
            end = datetime.fromisoformat(node_data['end_time_iso'])
            captions = {}
            text_emotions = {}

            for img_path in node_data['images']:
                caption = ""

                # Prefer bob-smith.json captions for Bob
                if user_id == "user_002":
                    caption = BOB_SMITH_CAPTIONS.get(img_path, "")

                # Fallback: use post caption for this image within episode range
                if not caption:
                    matching_post = next(
                        (
                            p for p in posts
                            if start <= p['timestamp'] < end
                            and p.get('image') == img_path
                            and isinstance(p.get('caption'), str)
                            and p['caption'].strip()
                        ),
                        None,
                    )
                    caption = matching_post['caption'] if matching_post else ""

                if caption:
                    captions[img_path] = caption
                    if _TEXT_SENTIMENT_AVAILABLE:
                        try:
                            text_emotions[img_path] = analyze_text_sentiment(caption)
                        except Exception as e:
                            text_emotions[img_path] = {
                                'positive': 0.33, 'negative': 0.33, 'neutral': 0.34,
                                'uncertainty_margin': 0.20,
                                'notes': f'Analysis error: {e}',
                                'disclaimer': 'Error during analysis.',
                            }

            if captions:
                node_data['captions'] = captions
            if text_emotions:
                node_data['text_emotions'] = text_emotions
        except Exception as e:
            # Pipeline stability improvement: skip this node on error
            app.logger.warning(f"WARNING: Failed to process captions/emotions for node: {e}")

    return jsonify(result)


@app.route('/api/cache-status/<user_id>')
def api_cache_status(user_id: str):
    posts = generate_sample_posts(user_id)
    timeline = build_timeline_from_posts(posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(hours=6))
    graph = build_narrative_graph(episodes) if episodes else None
    
    return jsonify({
        'user_id': user_id,
        'timeline_cached': cache.is_valid(timeline.fingerprint()),
        'graph_cached': cache.is_valid(graph.graph_id) if graph else False,
    })


@app.route('/api/text-emotion', methods=['POST'])
def api_text_emotion():
    """
    Analyze text for sentiment using the text-based ML model.
    Expects JSON body: {"text": "description text"}
    Returns positive/negative/neutral percentages.
    """
    if not _TEXT_SENTIMENT_AVAILABLE:
        return jsonify({
            'error': 'Text sentiment dependencies not installed',
            'message': _TEXT_SENTIMENT_ERROR,
        }), 500

    from flask import request
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    try:
        result = analyze_text_sentiment(data['text'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/perceptual-emotion/<path:image_path>')
def api_perceptual_emotion(image_path: str):
    """
    Analyze an image for perceptual emotion estimation using the latest CNN model.
    
    Uses the fer2013_mini_XCEPTION model for facial emotion detection.
    This endpoint demonstrates real-world perceptual uncertainty.
    """
    if not _PERCEPTUAL_DEPS_AVAILABLE:
        if _PERCEPTUAL_DEPS_ERROR:
            app.logger.warning("Perceptual emotion dependencies unavailable: %s", _PERCEPTUAL_DEPS_ERROR)
        return jsonify({
            'error': 'Required dependencies not installed',
            'message': 'Perceptual emotion analysis is currently unavailable.',
            'fallback': {
                'positive': 0.20,
                'neutral': 0.70,
                'negative': 0.10,
                'uncertainty_margin': 0.15,
                'disclaimer': 'This is a perceptual estimate, not emotional truth.',
            }
        }), 200
    
    try:
        from urllib.parse import unquote
        image_path = unquote(image_path)
        
        images_dir = Path(__file__).parent / "images"
        sample_dir = Path(__file__).parent / "sample-data"

        # Validate and resolve path against allowed roots
        full_path = None
        for root_dir in (images_dir, sample_dir):
            try:
                candidate = (root_dir / image_path).resolve()
                candidate.relative_to(root_dir.resolve())
            except ValueError:
                continue

            if candidate.exists():
                full_path = candidate
                break

        if full_path is None:
            return jsonify({'error': f'Image not found: {image_path}'}), 404
        
        img = Image.open(full_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        
        estimate = estimate_emotion_from_image(img_array)
        estimate['image_path'] = image_path
        
        return jsonify(estimate)
        
    except Exception:
        app.logger.exception("Failed to analyze perceptual emotion")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/compare-images')
def api_compare_images():
    """
    Compare emotion estimates between the two sample images using latest model.
    
    Uses the fer2013_mini_XCEPTION model for facial emotion detection.
    Demonstrates that different images produce different probability distributions.
    """
    if not _PERCEPTUAL_DEPS_AVAILABLE:
        return jsonify({
            'error': 'Dependencies not installed',
            'message': 'Perceptual emotion analysis is currently unavailable.'
        }), 500
    
    try:
        images_dir = Path(__file__).parent / "images"
        
        img_a = Image.open(images_dir / "download.jpeg").convert('RGB')
        img_b = Image.open(images_dir / "download (1).jpeg").convert('RGB')
        
        array_a = np.array(img_a).astype(np.float32) / 255.0
        array_b = np.array(img_b).astype(np.float32) / 255.0
        
        comparison = compare_image_estimates(array_a, array_b)
        comparison['image_a']['image_path'] = 'download.jpeg'
        comparison['image_b']['image_path'] = 'download (1).jpeg'
        
        return jsonify(comparison)
        
    except Exception:
        app.logger.exception("Failed to compare image emotion estimates")
        return jsonify({'error': 'An internal error occurred while comparing images.'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DREAMS Analytics Integration Server")
    print("="*60)
    print(f"Cache: {cache_dir}")
    print("\nOpen http://127.0.0.1:5001 in your browser")
    print("="*60 + "\n")
    
    debug_mode = os.getenv("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    app.run(debug=debug_mode, port=5001)
