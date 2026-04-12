import logging
import numpy as np

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional in lightweight test envs
    hdbscan = None

logger = logging.getLogger(__name__)

def get_vectors_and_metadata(doc):
    vectors = []
    metadata = []

    for sentiment in ['positive_keywords', 'negative_keywords']:
        for kw in doc.get(sentiment, []):
            vec = kw.get('embedding')  # Correct key here
            if vec:
                vectors.append(vec)
                metadata.append({
                    'keyword': kw.get('keyword'),
                    'sentiment': sentiment
                })

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

        vectors, metadata = get_vectors_and_metadata(doc)
        if len(vectors) < 2:
            logger.debug(f"Skipping user {user_id}: insufficient data ({len(vectors)} vectors)")
            continue  # Skip clustering if insufficient data

        logger.debug(f"Clustering user {user_id}: vectors shape {vectors.shape}")
        logger.debug(f"Sample vectors for user {user_id} (first 5): {vectors[:5]}")

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        cluster_labels = clusterer.fit_predict(vectors)

        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_count = np.count_nonzero(cluster_labels == -1)
        logger.debug(f"HDBSCAN produced {unique_clusters} clusters for user {user_id} ({noise_count} noise points)")

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

    logger.info("Clustering complete for all users")
