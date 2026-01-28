#!/usr/bin/env python3

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, render_template_string, redirect, url_for
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
    _PERCEPTUAL_DEPS_ERROR = str(e)
    print(f"⚠️  WARNING: Perceptual emotion analysis dependencies not available: {e}")
    print("   The /api/perceptual-emotion and /api/compare-images endpoints will return fallback data.")

# Simulated users with emotion data
SAMPLE_USERS = {
    "user_001": "Alice Johnson",
    "user_002": "Bob Smith", 
    "user_003": "Carol Williams",
}


# Alice's special images
ALICE_IMAGES = [
    "/static/images/download.jpeg",
    "/static/images/download%20(1).jpeg",
    "/static/images/download%20(2).jpeg"
]

def generate_sample_posts(user_id: str) -> list:
    """Generate sample posts with emotions for a user."""
    random.seed(hash(user_id))  # Deterministic per user
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    emotions = ["Happiness", "Sadness", "Fear", "Anger", "Disgust", "Surprise"]
    posts = []
    
    # Generate 20-40 posts over 30 days
    num_posts = random.randint(20, 40)
    
    for i in range(num_posts):
        day_offset = random.randint(0, 29)
        hour_offset = random.randint(0, 23)
        minute_offset = random.randint(0, 59)
        
        timestamp = base_time + timedelta(days=day_offset, hours=hour_offset, minutes=minute_offset)
        
        # Bias towards different emotions for different users
        if "001" in user_id:
            weights = [0.4, 0.1, 0.05, 0.05, 0.05, 0.35]  # More Happiness and Surprise
        elif "002" in user_id:
            weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.2]  # Balanced
        else:
            weights = [0.1, 0.3, 0.15, 0.2, 0.15, 0.1]  # More Sadness and Anger
        
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
    
    return posts


def build_timeline_from_posts(user_posts: list, user_id: str) -> EmotionTimeline:
    """Convert posts to EmotionTimeline."""
    events = []
    for post in user_posts:
        sentiment = post.get('sentiment', {})
        events.append(EmotionEvent(
            timestamp=post['timestamp'],
            emotion_label=sentiment.get('label', 'Surprise'),
            score=sentiment.get('score'),
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
        <h1 class="text-center mb-4">📊 DREAMS Analytics Demo</h1>
        <p class="text-center text-muted mb-5">PR-6: Canonical Identity, Serialization, Persistence, and Frontend Contract</p>
        
        <div class="row justify-content-center">
            {% for user_id, name in users.items() %}
            <div class="col-md-4 mb-4">
                <div class="card p-4 text-center">
                    <h5>{{ name }}</h5>
                    <p class="text-muted small">{{ user_id }}</p>
                    <div class="d-grid gap-2">
                        <a href="{{ url_for('narrative_view', user_id=user_id) }}" class="btn btn-narrative text-white">📊 View Narrative</a>
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
        .timeline-bar { height: 60px; background: #1a1a2e; border-radius: 0.5rem; position: relative; overflow: hidden; margin: 1rem 0; }
        .timeline-event { position: absolute; height: 100%; min-width: 4px; border-radius: 2px; }
        .timeline-event.Happiness { background: #4ad974; }
        .timeline-event.Sadness { background: #4a90d9; }
        .timeline-event.Fear { background: #9b59b6; }
        .timeline-event.Anger { background: #d94a4a; }
        .timeline-event.Disgust { background: #27ae60; }
        .timeline-event.Surprise { background: #f39c12; }
        .fingerprint { font-family: monospace; font-size: 0.8rem; color: #4a90d9; background: #1a1a2e; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
        .btn-back { background: #333; border: none; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); }
        .modal.show { display: flex; align-items: center; justify-content: center; }
        .modal-content { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 2px solid #4a90d9; border-radius: 1rem; padding: 2rem; max-width: 700px; width: 90%; position: relative; }
        .modal-close { position: absolute; top: 1rem; right: 1rem; font-size: 2rem; color: #e0e0e0; cursor: pointer; background: none; border: none; }
        .modal-close:hover { color: #4a90d9; }
        #emotionChart { max-height: 300px; }
        .perceptual-badge { display: inline-block; background: linear-gradient(135deg, #ff6b6b 0%, #ffa502 100%); color: #000; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.75rem; font-weight: bold; margin-bottom: 1rem; }
        .disclaimer { font-size: 0.75rem; color: #888; font-style: italic; margin-top: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 0.5rem; }
        .uncertainty-bar { height: 8px; background: #333; border-radius: 4px; margin: 0.5rem 0; overflow: hidden; }
        .uncertainty-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
        .prob-row { display: flex; align-items: center; margin: 0.5rem 0; }
        .prob-label { width: 80px; font-weight: bold; }
        .prob-bar-container { flex: 1; margin: 0 1rem; }
        .prob-bar { height: 24px; border-radius: 12px; transition: width 0.5s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 0.5rem; font-size: 0.8rem; color: white; }
        .prob-Happiness { background: linear-gradient(90deg, #2e7d32 0%, #4ad974 100%); }
        .prob-Sadness { background: linear-gradient(90deg, #357abd 0%, #4a90d9 100%); }
        .prob-Fear { background: linear-gradient(90deg, #8e44ad 0%, #9b59b6 100%); }
        .prob-Anger { background: linear-gradient(90deg, #bd3535 0%, #d94a4a 100%); }
        .prob-Disgust { background: linear-gradient(90deg, #1e8449 0%, #27ae60 100%); }
        .prob-Surprise { background: linear-gradient(90deg, #d68910 0%, #f39c12 100%); }
        .image-preview { max-width: 150px; max-height: 150px; border-radius: 0.5rem; object-fit: cover; margin-right: 1.5rem; }
        .modal-flex { display: flex; align-items: flex-start; }
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2>📊 Temporal Narrative: {{ user_id }}</h2>
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
            <h4>Emotion Timeline</h4>
            <div class="timeline-bar" id="timeline-bar"></div>
            <div class="d-flex justify-content-between text-muted small">
                <span id="timeline-start">-</span>
                <span id="timeline-end">-</span>
            </div>
        </div>

        <div class="section">
            <h4>Episode Network <small class="text-muted">(click images for perceptual analysis)</small></h4>
            <div class="graph-container" id="graph-container"><div class="text-muted">Loading...</div></div>
        </div>

        <div class="section">
            <h4>Connections</h4>
            <div id="edges-container" class="text-center"><span class="text-muted">Loading...</span></div>
        </div>
    </div>
    
    <div id="emotionModal" class="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <div class="text-center">
                <span class="perceptual-badge">🔬 PERCEPTUAL ESTIMATE</span>
            </div>
            <h4 id="modal-title" class="mb-4 text-center">Emotion Analysis</h4>
            
            <div class="modal-flex">
                <img id="modal-image" class="image-preview" src="" alt="Analyzed image">
                <div style="flex: 1;">
                    <div id="prob-display">
                        <div class="prob-row">
                            <span class="prob-label" style="color: #4ad974;">Happiness</span>
                            <div class="prob-bar-container">
                                <div id="prob-bar-Happiness" class="prob-bar prob-Happiness" style="width: 0%;">0%</div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <span class="prob-label" style="color: #4a90d9;">Sadness</span>
                            <div class="prob-bar-container">
                                <div id="prob-bar-Sadness" class="prob-bar prob-Sadness" style="width: 0%;">0%</div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <span class="prob-label" style="color: #9b59b6;">Fear</span>
                            <div class="prob-bar-container">
                                <div id="prob-bar-Fear" class="prob-bar prob-Fear" style="width: 0%;">0%</div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <span class="prob-label" style="color: #d94a4a;">Anger</span>
                            <div class="prob-bar-container">
                                <div id="prob-bar-Anger" class="prob-bar prob-Anger" style="width: 0%;">0%</div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <span class="prob-label" style="color: #27ae60;">Disgust</span>
                            <div class="prob-bar-container">
                                <div id="prob-bar-Disgust" class="prob-bar prob-Disgust" style="width: 0%;">0%</div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <span class="prob-label" style="color: #f39c12;">Surprise</span>
                            <div class="prob-bar-container">
                                <div id="prob-bar-Surprise" class="prob-bar prob-Surprise" style="width: 0%;">0%</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mt-3">
                        <strong>Uncertainty Margin:</strong>
                        <div class="uncertainty-bar">
                            <div id="uncertainty-fill" class="uncertainty-fill" style="width: 0%; background: #ffa502;"></div>
                        </div>
                        <span id="uncertainty-text" class="small text-muted">0%</span>
                    </div>
                    
                    <div id="notes-text" class="small text-muted mt-2"></div>
                </div>
            </div>
            
            <div class="mt-4">
                <canvas id="emotionChart"></canvas>
            </div>
            
            <div class="disclaimer">
                ⚠️ <strong>Not emotional truth.</strong> This is a probabilistic perceptual estimate demonstrating uncertainty. 
                No confidence is ever 100%. This does not feed into DREAMS structural analytics.
            </div>
        </div>
    </div>

    <script>
        const userId = "{{ user_id }}";
        let allEpisodeData = [];
        let currentChart = null;
        
        async function loadData() {
            const timelineRes = await fetch(`/api/timeline/${userId}`);
            const timeline = await timelineRes.json();
            
            document.getElementById('stat-events').textContent = timeline.event_count;
            document.getElementById('graph-fingerprint').textContent = `Fingerprint: ${timeline.fingerprint}`;
            renderTimeline(timeline);

            const payloadRes = await fetch(`/api/frontend-payload/${userId}`);
            const payload = await payloadRes.json();
            
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
        }
        
        function renderTimeline(timeline) {
            const container = document.getElementById('timeline-bar');
            container.innerHTML = '';
            if (!timeline.events || !timeline.events.length) return;
            
            const startTime = new Date(timeline.temporal_bounds.start).getTime();
            const endTime = new Date(timeline.temporal_bounds.end).getTime();
            const duration = endTime - startTime || 1;
            
            document.getElementById('timeline-start').textContent = new Date(startTime).toLocaleDateString();
            document.getElementById('timeline-end').textContent = new Date(endTime).toLocaleDateString();
            
            timeline.events.forEach(event => {
                const pos = ((new Date(event.timestamp).getTime() - startTime) / duration) * 100;
                const div = document.createElement('div');
                div.className = `timeline-event ${event.emotion_label}`;
                div.style.left = `${pos}%`;
                div.style.width = `${Math.max(1, 100 / timeline.events.length)}%`;
                div.title = `${event.emotion_label} - ${new Date(event.timestamp).toLocaleString()}`;
                container.appendChild(div);
            });
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
                        const imgPath = imgSrc.replace('/static/images/', '');
                        return `<img src="${imgSrc}" alt="Episode ${node.index + 1} Image ${imgIdx + 1}" 
                            style="width: ${node.images.length > 1 ? '48%' : '100%'}; height: ${node.images.length > 1 ? '80px' : '120px'}; object-fit: cover; border-radius: 8px; cursor: pointer;" 
                            onclick="showPerceptualAnalysis('${imgPath}', '${imgSrc}', ${idx})">`;
                    }).join('');
                    imageHTML = `<div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 0.5rem;">${imagesContainer}</div>`;
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
        
        async function showPerceptualAnalysis(imagePath, imageSrc, episodeIndex) {
            document.getElementById('modal-title').textContent = `Episode ${episodeIndex + 1} - Perceptual Analysis`;
            document.getElementById('modal-image').src = imageSrc;
            document.getElementById('emotionModal').classList.add('show');
            
            // Destroy previous chart immediately
            if (currentChart) {
                currentChart.destroy();
                currentChart = null;
            }
            
            // Reset displays
            document.getElementById('prob-bar-Happiness').style.width = '0%';
            document.getElementById('prob-bar-Happiness').textContent = '0%';
            document.getElementById('prob-bar-Sadness').style.width = '0%';
            document.getElementById('prob-bar-Sadness').textContent = '0%';
            document.getElementById('prob-bar-Fear').style.width = '0%';
            document.getElementById('prob-bar-Fear').textContent = '0%';
            document.getElementById('prob-bar-Anger').style.width = '0%';
            document.getElementById('prob-bar-Anger').textContent = '0%';
            document.getElementById('prob-bar-Disgust').style.width = '0%';
            document.getElementById('prob-bar-Disgust').textContent = '0%';
            document.getElementById('prob-bar-Surprise').style.width = '0%';
            document.getElementById('prob-bar-Surprise').textContent = '0%';
            document.getElementById('uncertainty-fill').style.width = '0%';
            document.getElementById('notes-text').textContent = 'Analyzing...';
            
            try {
                const response = await fetch(`/api/perceptual-emotion/${encodeURIComponent(imagePath)}`);
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('notes-text').textContent = 'Error: ' + data.error;
                    return;
                }
                
                // Animate probability bars - use 6 basic emotions
                const emotions = ['Happiness', 'Sadness', 'Fear', 'Anger', 'Disgust', 'Surprise'];
                emotions.forEach((emotion, idx) => {
                    setTimeout(() => {
                        const value = data[emotion] || 0;
                        document.getElementById(`prob-bar-${emotion}`).style.width = `${value * 100}%`;
                        document.getElementById(`prob-bar-${emotion}`).textContent = `${(value * 100).toFixed(1)}%`;
                    }, 100 * (idx + 1));
                });
                
                // Uncertainty bar
                setTimeout(() => {
                    const uncertainty = data.uncertainty_margin * 100;
                    document.getElementById('uncertainty-fill').style.width = `${uncertainty * 5}%`;
                    document.getElementById('uncertainty-text').textContent = `±${uncertainty.toFixed(1)}%`;
                }, 700);
                
                document.getElementById('notes-text').textContent = data.notes;
                
                // Create chart with 6 basic emotions - wait a bit for animations to start
                setTimeout(() => {
                    const ctx = document.getElementById('emotionChart').getContext('2d');
                    currentChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: ['Happiness', 'Sadness', 'Fear', 'Anger', 'Disgust', 'Surprise'],
                            datasets: [{
                                data: [data.Happiness || 0, data.Sadness || 0, data.Fear || 0, data.Anger || 0, data.Disgust || 0, data.Surprise || 0],
                                backgroundColor: ['#4ad974', '#4a90d9', '#9b59b6', '#d94a4a', '#27ae60', '#f39c12'],
                                borderColor: ['#2e7d32', '#357abd', '#8e44ad', '#bd3535', '#1e8449', '#d68910'],
                                borderWidth: 2
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: {
                                legend: { labels: { color: '#e0e0e0', font: { size: 12 } } },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const percentage = (context.parsed * 100).toFixed(1);
                                            return `${context.label}: ${percentage}%`;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }, 100);
                
            } catch (error) {
                document.getElementById('notes-text').textContent = 'Error: ' + error.message;
            }
        }
        
        function closeModal() {
            document.getElementById('emotionModal').classList.remove('show');
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
    return render_template_string(NARRATIVE_TEMPLATE, user_id=user_id)


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
    from flask import send_from_directory
    images_dir = Path(__file__).parent / "images"
    
    # Validate filename to prevent path traversal
    try:
        safe_path = (images_dir / filename).resolve()
        safe_path.relative_to(images_dir.resolve())
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400
    
    return send_from_directory(images_dir, filename)

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
    posts_by_timestamp = {p['timestamp']: p for p in posts}
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


@app.route('/api/perceptual-emotion/<path:image_path>')
def api_perceptual_emotion(image_path: str):
    """
    Analyze an image for perceptual emotion estimation using the latest CNN model.
    
    Uses the fer2013_mini_XCEPTION model for facial emotion detection.
    Returns 6 basic emotions: Happiness, Sadness, Fear, Anger, Disgust, Surprise.
    This endpoint demonstrates real-world perceptual uncertainty.
    """
    if not _PERCEPTUAL_DEPS_AVAILABLE:
        return jsonify({
            'error': 'Required dependencies not installed',
            'message': _PERCEPTUAL_DEPS_ERROR,
            'fallback': {
                'Happiness': 0.20,
                'Sadness': 0.15,
                'Fear': 0.10,
                'Anger': 0.15,
                'Disgust': 0.10,
                'Surprise': 0.30,
                'uncertainty_margin': 0.15,
                'disclaimer': 'This is a perceptual estimate using 6 basic emotions.',
            }
        }), 200
    
    try:
        from urllib.parse import unquote
        image_path = unquote(image_path)
        
        images_dir = Path(__file__).parent / "images"
        
        # Validate path to prevent directory traversal attacks
        try:
            full_path = (images_dir / image_path).resolve()
            full_path.relative_to(images_dir.resolve())
        except ValueError:
            return jsonify({'error': 'Invalid path: access denied'}), 400
        
        if not full_path.exists():
            return jsonify({'error': f'Image not found: {image_path}'}), 404
        
        img = Image.open(full_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        
        estimate = estimate_emotion_from_image(img_array)
        estimate['image_path'] = image_path
        
        return jsonify(estimate)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-images')
def api_compare_images():
    """
    Compare emotion estimates between the two sample images using latest model.
    
    Uses the fer2013_mini_XCEPTION model for facial emotion detection.
    Returns 6 basic emotions: Happiness, Sadness, Fear, Anger, Disgust, Surprise.
    Demonstrates that different images produce different probability distributions.
    """
    if not _PERCEPTUAL_DEPS_AVAILABLE:
        return jsonify({
            'error': 'Dependencies not installed',
            'message': _PERCEPTUAL_DEPS_ERROR
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DREAMS Analytics Integration Server")
    print("="*60)
    print(f"Cache: {cache_dir}")
    print("\nOpen http://127.0.0.1:5001 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001)
