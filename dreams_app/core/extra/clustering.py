import logging
import numpy as np

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional in lightweight test envs
    hdbscan = None

logger = logging.getLogger(__name__)

def _embed_keywords(keywords):
    """Embed keyword text with the model used for keyword extraction.

    Returns None when the model is not available.
    """
    from dreams_app.core.extra.keywords import _get_keyword_model

    model = _get_keyword_model()
    if model is None:
        return None
    return model.encode(keywords)


def get_vectors_and_metadata(doc, embed=None):
    """Collect one vector per keyword of a user's document.

    upload_post stores keywords without their embedding, so keywords that have no
    stored embedding are embedded on demand with ``embed`` (a function taking a list
    of keyword strings and returning one vector each). Without ``embed`` they are
    left out.
    """
    vectors = []
    metadata = []
    to_embed = []

    for sentiment in ['positive_keywords', 'negative_keywords']:
        for kw in doc.get(sentiment, []):
            entry = {
                'keyword': kw.get('keyword'),
                'sentiment': sentiment
            }
            vec = kw.get('embedding')  # Correct key here
            if vec:
                vectors.append(vec)
                metadata.append(entry)
            elif embed is not None and entry['keyword']:
                to_embed.append(entry)

    if to_embed:
        embedded = embed([entry['keyword'] for entry in to_embed])
        if embedded is not None:
            vectors.extend(embedded)
            metadata.extend(to_embed)

    return np.array(vectors), metadata

def cluster_keywords_for_all_users(keywords_collection):
    if hdbscan is None:
        logger.warning("hdbscan is not installed; skipping keyword clustering")
        return

    all_users = keywords_collection.find({})

    for doc in all_users:
        user_id = doc.get('user_id')
        if not user_id:
            continue

        vectors, metadata = get_vectors_and_metadata(doc, embed=_embed_keywords)
        if len(vectors) < 2:
            user_id = doc.get('user_id', 'unknown')
            logger.debug(f"Skipping user {user_id}: insufficient data ({len(vectors)} vectors)")
            continue  # Skip clustering if insufficient data

        user_id = doc.get('user_id', 'unknown')
        logger.debug(f"Clustering user {user_id}: vectors shape {vectors.shape}")
        logger.debug(f"Sample vectors for user {user_id} (first 5): {vectors[:5]}")

        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
            cluster_labels = clusterer.fit_predict(vectors)

            unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            noise_count = np.sum(cluster_labels == -1)
            logger.info(f"HDBSCAN produced {unique_clusters} clusters for user {user_id} ({noise_count} noise points)")

            clustered_result = []
            for i, label in enumerate(cluster_labels):
                clustered_result.append({
                    'keyword': metadata[i]['keyword'],
                    'sentiment': metadata[i]['sentiment'],
                    'cluster': int(label) if label != -1 else 'noise'
                })

            # Store result back in document
            keywords_collection.update_one(
                {'user_id': user_id},
                {'$set': {'clustered_keywords': clustered_result}}
            )
        except Exception:
            logger.exception("Clustering failed for user_id=%s; skipping this user", user_id)
            continue

    logger.info("Clustering complete for all users")
