from flask import current_app
import numpy as np
import hdbscan

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
    mongo = current_app.mongo
    keywords_collection = mongo['keywords']

    all_users = keywords_collection.find({})

    for doc in all_users:
        user_id = doc.get('user_id')
        if not user_id:
            continue

        vectors, metadata = get_vectors_and_metadata(doc)
        if len(vectors) < 2:
            continue  # Skip clustering if insufficient data

        # Debug: Print the shape of the vectors array to check its dimensions
        print(f"Shape of vectors array: {vectors.shape}")
        # Debug: Print the first few vectors to inspect their values
        print(f"First 5 vectors: {vectors[:5]}")

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        cluster_labels = clusterer.fit_predict(vectors)

        # Debug: Print the cluster labels to see how the data is being clustered
        print(f"Cluster labels: {cluster_labels}")

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

    print("All users clustered.")
