# dreamsApp/app/analytics/routes.py

"""
API routes for graph-based narrative structure analysis.

Endpoint:
    GET /api/analytics/graph-metrics/<user_id>
        Returns quantitative structural metrics for the user's emotional
        narrative graph (centrality, transitions, cycles, etc.).
"""

import logging
import re

from flask import jsonify, current_app
from flask_login import login_required
from . import bp

logger = logging.getLogger(__name__)

# Security: only allow reasonable user_id values (letters, digits, _.-@)
_USER_ID_RE = re.compile(r'^[\w.\-@]{1,64}$')


@bp.route('/graph-metrics/<string:user_id>', methods=['GET'])
@login_required
def graph_metrics(user_id: str):
    """
    Compute and return structural graph metrics for a user's emotional narrative.

    Response 200:
        JSON with ``user_id`` and ``metrics`` (graph_summary, node_metrics,
        pattern_analysis).
    Response 404:
        No posts found for the given user_id.
    Response 500:
        Internal error during analysis (details logged server-side only).
    """
    try:
        if not _USER_ID_RE.match(user_id):
            logger.warning(f"Invalid user_id format: {user_id}")
            return jsonify({'error': 'Invalid user_id format'}), 400

        import sqlite3
        from dreamsApp.core.database import db_manager
        
        with sqlite3.connect(db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute("SELECT * FROM posts WHERE user_id = ? ORDER BY timestamp ASC", (user_id,)).fetchall()
            
            user_posts = []
            for r in rows:
                post = dict(r)
                post['_id'] = str(post['id'])
                post['sentiment'] = {'label': post['sentiment_label'], 'score': post['sentiment_score']}
                user_posts.append(post)

        if not user_posts:
            return jsonify({
                'error': 'No posts found for user',
                'user_id': user_id,
            }), 404

        # Delegate heavy graph execution to the global pipeline
        metrics = current_app.dreams_pipeline.generate_narrative_metrics(user_id, user_posts)

        return jsonify({
            'user_id': user_id,
            'metrics': metrics,
        })

    except Exception:
        # Log the full traceback server-side; never expose it to the client.
        logger.exception(
            "Failed to compute graph metrics for user_id=%s", user_id
        )
        return jsonify({'error': 'Failed to compute graph metrics'}), 500
