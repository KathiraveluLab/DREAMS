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

        mongo = current_app.mongo['posts']

        # Fetch posts sorted chronologically
        user_posts = list(
            mongo.find({'user_id': user_id}).sort('timestamp', 1)
        )

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
