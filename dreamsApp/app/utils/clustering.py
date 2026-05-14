import logging

from flask import current_app
import numpy as np
import hdbscan

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

def cluster_keywords_for_all_users():
    """Cluster keywords for all users using HDBSCAN.

    Pipeline stability improvement: individual user failures are
    logged and skipped — never crash the whole clustering run.
    """
    mongo = current_app.mongo
    keywords_collection = mongo['keywords']

    all_users = keywords_collection.find({})

    for doc in all_users:
        user_id = doc.get('user_id')
        if not user_id:
            continue

        vectors, metadata = get_vectors_and_metadata(doc)
        if len(vectors) < 2:
            logger.info("Skipping clustering for user_id=%s — insufficient data (%d vectors)", user_id, len(vectors))
            continue  # Skip clustering if insufficient data

        # Pipeline stability improvement: HDBSCAN failure handling
        try:
            logger.info("Clustering %d vectors for user_id=%s", len(vectors), user_id)

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
            cluster_labels = clusterer.fit_predict(vectors)

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
        except Exception as e:
            logger.error("Clustering failed for user_id=%s: %s — skipping this user", user_id, e)
            continue

    logger.info("All users clustered.")
