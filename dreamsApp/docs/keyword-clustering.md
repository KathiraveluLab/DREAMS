# Keyword Clustering – DREAMS Platform

This document describes the data structure and processing logic for the keyword clustering module in the DREAMS system. The goal is to semantically group caption-derived keywords and associate them with sentiment and user identity for narrative analysis.

---

##  Input Source

- User-submitted **caption** (free text)
- Sentiment-polarity classification (e.g., positive/negative)
- Generated image caption (BLIP) — *optional enhancement*

---

##  Collection: `keywords`

Each document in the `keywords` collection stores the extracted and clustered keywords for a single user.

| Field                | Type            | Description                                 |
|----------------------|-----------------|---------------------------------------------|
| `_id`                | `ObjectId`      | MongoDB document ID                         |
| `user_id`            | `string`        | Associated user ID                          |
| `positive_keywords`  | `array<object>` | Clustered keywords with positive sentiment  |
| `negative_keywords`  | `array<object>` | Clustered keywords with negative sentiment  |

---

##  Keyword Object Structure

Each keyword is stored with semantic metadata:

```json
{
    "keyword": "a large tower",
    "vector": [0.15, -0.73, 0.48, ...],
    "cluster": 2
}
```

| Field     | Type           | Description                                      |
|-----------|----------------|--------------------------------------------------|
| keyword   | `string`       | Extracted noun phrase or keyword                 |
| vector    | `array<float>` | Optional embedding (e.g., from SBERT or FastText)|
| cluster   | `int/string`   | Cluster ID (semantic group)                      |

---

##  Processing Pipeline

1. **Text Preprocessing**
     - Tokenization
     - Noun phrase extraction (spaCy or similar)
     - *Optional:* Filtering stop words and noise

2. **Embedding Generation**
     - Use Sentence-BERT or other embedding models
     - Normalize embeddings for clustering

3. **Sentiment Classification**
     - Caption is labeled as positive, negative, or neutral
     - Determines which list (`positive_keywords`/`negative_keywords`) to update

4. **Clustering**
     - Apply unsupervised clustering (e.g., KMeans, DBSCAN, HDBSCAN)
     - Assign cluster labels to each keyword
     - Store along with vector in the appropriate sentiment group

5. **Storage in MongoDB**
     - `$push` keyword entries into `positive_keywords` or `negative_keywords`
     - If new user, initialize opposite group as empty for structural consistency

---

##  Example Document

```json
{
    "_id": ObjectId("66c12df..."),
    "user_id": "111",
    "positive_keywords": [
        {
            "keyword": "a person",
            "vector": [...],
            "cluster": 0
        },
        {
            "keyword": "a large tower",
            "vector": [...],
            "cluster": 1
        }
    ],
    "negative_keywords": [
        {
            "keyword": "a broken wall",
            "vector": [...],
            "cluster": 2
        }
    ]
}
```

---

##  Usage in Analytics

Clusters help in:

- Identifying emerging themes over time
- Filtering noise vs. meaningful content
- Mapping emotional trajectory through cluster shifts

**Visualizations:**

- Bubble charts (keywords by cluster)
- Temporal cluster maps (change over time)
- Sentiment-cluster heatmaps

---

##  Notes

- Admins retrieve clusters via the dashboard for user analysis.
- Cluster structure is updated incrementally as new captions are posted.
- Neutral sentiment is ignored or optionally stored in a third category.

---

##  Future Extensions

- Weight keywords by TF-IDF or caption length
- Track `cluster_drift` over time
- Add `created_at` timestamp for each keyword entry
- Implement cluster merging or renaming based on LLM feedback

---

> **Keyword clustering enables scalable, interpretable emotional trend mining across visual narratives, empowering meaningful insights for social impact.**
