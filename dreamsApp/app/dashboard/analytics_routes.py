from datetime import datetime, timedelta
from pathlib import Path
from flask import jsonify, current_app, request, render_template
from flask_login import login_required
import tempfile

from . import bp

# Import analytics modules
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


def get_cache() -> StructuralCache:
    """Get or create the structural cache."""
    if not hasattr(current_app, '_analytics_cache'):
        cache_dir = Path(current_app.instance_path) / "analytics_cache"
        store = ContentAddressedStore(cache_dir)
        current_app._analytics_cache = StructuralCache(store)
    return current_app._analytics_cache


def build_timeline_from_posts(user_posts: list, user_id: str) -> EmotionTimeline:
    """Convert MongoDB posts to EmotionTimeline."""
    events = []
    for post in user_posts:
        sentiment = post.get('sentiment', {})
        events.append(EmotionEvent(
            timestamp=post['timestamp'],
            emotion_label=sentiment.get('label', 'neutral'),
            score=sentiment.get('score'),
            source_id=str(post.get('_id', '')),
            metadata={'caption': post.get('caption', '')[:100]} if post.get('caption') else None,
        ))
    
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    return EmotionTimeline(subject_id=user_id, events=tuple(events_sorted))


@bp.route('/narrative/<string:user_id>')
@login_required
def narrative_view(user_id: str):
    """Render the narrative graph visualization page."""
    return render_template('dashboard/narrative.html', user_id=user_id)


@bp.route('/api/timeline/<string:user_id>')
@login_required
def get_user_timeline(user_id: str):
    """Get emotion timeline for a user with structural fingerprint."""
    mongo = current_app.mongo['posts']
    user_posts = list(mongo.find({'user_id': user_id}).sort('timestamp', 1))
    
    if not user_posts:
        return jsonify({'error': 'No posts found for user'}), 404
    
    timeline = build_timeline_from_posts(user_posts, user_id)
    cache = get_cache()
    
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
            {
                'timestamp': e.timestamp.isoformat(),
                'emotion_label': e.emotion_label,
                'score': e.score,
            }
            for e in timeline.events
        ]
    })


@bp.route('/api/episodes/<string:user_id>')
@login_required
def get_user_episodes(user_id: str):
    """Get temporal episodes for a user with stable IDs."""
    gap_threshold_minutes = request.args.get('gap_threshold', 30, type=int)
    
    mongo = current_app.mongo['posts']
    user_posts = list(mongo.find({'user_id': user_id}).sort('timestamp', 1))
    
    if not user_posts:
        return jsonify({'error': 'No posts found for user'}), 404
    
    timeline = build_timeline_from_posts(user_posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=gap_threshold_minutes))
    
    cache = get_cache()
    
    episodes_data = []
    for ep in episodes:
        payload = EpisodeSerializer.serialize(ep)
        cache.put(payload)
        
        episodes_data.append({
            'episode_id': ep.episode_id,
            'start_time': ep.start_time.isoformat(),
            'end_time': ep.end_time.isoformat(),
            'duration_seconds': ep.duration(),
            'event_count': len(ep),
            'cached': cache.is_valid(ep.episode_id),
            'events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'emotion_label': e.emotion_label,
                    'score': e.score,
                }
                for e in ep.events
            ]
        })
    
    return jsonify({
        'user_id': user_id,
        'schema_version': SCHEMA_VERSION,
        'gap_threshold_minutes': gap_threshold_minutes,
        'episode_count': len(episodes_data),
        'episodes': episodes_data,
    })


@bp.route('/api/narrative-graph/<string:user_id>')
@login_required
def get_narrative_graph(user_id: str):
    """Get temporal narrative graph for a user with stable IDs."""
    gap_threshold_minutes = request.args.get('gap_threshold', 30, type=int)
    adjacency_threshold_minutes = request.args.get('adjacency_threshold', 60, type=int)
    include_disjoint = request.args.get('include_disjoint', 'false').lower() == 'true'
    
    mongo = current_app.mongo['posts']
    user_posts = list(mongo.find({'user_id': user_id}).sort('timestamp', 1))
    
    if not user_posts:
        return jsonify({'error': 'No posts found for user'}), 404
    
    timeline = build_timeline_from_posts(user_posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=gap_threshold_minutes))
    
    if not episodes:
        return jsonify({
            'user_id': user_id,
            'graph_id': None,
            'node_count': 0,
            'edge_count': 0,
            'nodes': [],
            'edges': [],
        })
    
    graph = build_narrative_graph(
        episodes,
        adjacency_threshold=timedelta(minutes=adjacency_threshold_minutes),
        include_disjoint_edges=include_disjoint
    )
    
    cache = get_cache()
    payload = TemporalNarrativeGraphSerializer.serialize(graph)
    cache.put(payload)
    
    return jsonify({
        'user_id': user_id,
        'schema_version': SCHEMA_VERSION,
        'graph_id': graph.graph_id,
        'cached': cache.is_valid(graph.graph_id),
        'node_count': graph.node_count(),
        'edge_count': graph.edge_count(),
        'parameters': {
            'gap_threshold_minutes': gap_threshold_minutes,
            'adjacency_threshold_minutes': adjacency_threshold_minutes,
            'include_disjoint': include_disjoint,
        },
        'nodes': [
            {
                'index': i,
                'episode_id': node.episode_id,
                'start_time': node.start_time.isoformat(),
                'end_time': node.end_time.isoformat(),
                'duration_seconds': node.duration(),
                'event_count': len(node),
            }
            for i, node in enumerate(graph.nodes)
        ],
        'edges': [
            {
                'source_index': edge.source_index,
                'target_index': edge.target_index,
                'relation': edge.relation.value,
            }
            for edge in graph.edges
        ],
    })


@bp.route('/api/frontend-payload/<string:user_id>')
@login_required
def get_frontend_graph_payload(user_id: str):
    """Get frontend-ready graph payload with stable IDs and deterministic ordering."""
    gap_threshold_minutes = request.args.get('gap_threshold', 30, type=int)
    adjacency_threshold_minutes = request.args.get('adjacency_threshold', 60, type=int)
    
    mongo = current_app.mongo['posts']
    user_posts = list(mongo.find({'user_id': user_id}).sort('timestamp', 1))
    
    if not user_posts:
        return jsonify({'error': 'No posts found for user'}), 404
    
    timeline = build_timeline_from_posts(user_posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=gap_threshold_minutes))
    
    if not episodes:
        return jsonify({
            'user_id': user_id,
            'schema_version': SCHEMA_VERSION,
            'graph_id': '',
            'nodes': [],
            'edges': [],
            'node_count': 0,
            'edge_count': 0,
        })
    
    graph = build_narrative_graph(
        episodes,
        adjacency_threshold=timedelta(minutes=adjacency_threshold_minutes)
    )
    
    frontend_payload = build_frontend_payload(graph)
    
    result = frontend_payload.to_dict()
    result['user_id'] = user_id
    
    return jsonify(result)


@bp.route('/api/cache-status/<string:user_id>')
@login_required
def get_user_cache_status(user_id: str):
    """Check cache status for a user's analytics data."""
    gap_threshold_minutes = request.args.get('gap_threshold', 30, type=int)
    
    mongo = current_app.mongo['posts']
    user_posts = list(mongo.find({'user_id': user_id}).sort('timestamp', 1))
    
    if not user_posts:
        return jsonify({'error': 'No posts found for user'}), 404
    
    timeline = build_timeline_from_posts(user_posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=gap_threshold_minutes))
    graph = build_narrative_graph(episodes) if episodes else None
    
    cache = get_cache()
    
    return jsonify({
        'user_id': user_id,
        'timeline': {
            'fingerprint': timeline.fingerprint(),
            'cached': cache.is_valid(timeline.fingerprint()),
        },
        'episodes': [
            {
                'episode_id': ep.episode_id,
                'cached': cache.is_valid(ep.episode_id),
            }
            for ep in episodes
        ],
        'graph': {
            'graph_id': graph.graph_id if graph else None,
            'cached': cache.is_valid(graph.graph_id) if graph else False,
        } if graph else None,
    })


@bp.route('/api/invalidate-cache/<string:user_id>', methods=['POST'])
@login_required
def invalidate_user_cache(user_id: str):
    """Invalidate cached analytics data for a user."""
    gap_threshold_minutes = request.args.get('gap_threshold', 30, type=int)
    
    mongo = current_app.mongo['posts']
    user_posts = list(mongo.find({'user_id': user_id}).sort('timestamp', 1))
    
    if not user_posts:
        return jsonify({'error': 'No posts found for user'}), 404
    
    timeline = build_timeline_from_posts(user_posts, user_id)
    episodes = segment_timeline_to_episodes(timeline, timedelta(minutes=gap_threshold_minutes))
    graph = build_narrative_graph(episodes) if episodes else None
    
    cache = get_cache()
    invalidated = []
    
    if cache.invalidate(timeline.fingerprint()):
        invalidated.append(f"timeline:{timeline.fingerprint()}")
    
    for ep in episodes:
        if cache.invalidate(ep.episode_id):
            invalidated.append(f"episode:{ep.episode_id}")
    
    if graph and cache.invalidate(graph.graph_id):
        invalidated.append(f"graph:{graph.graph_id}")
    
    return jsonify({
        'user_id': user_id,
        'invalidated_count': len(invalidated),
        'invalidated': invalidated,
    })
