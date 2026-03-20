import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from dreamsApp.core.config import PipelineConfig
from dreamsApp.core.sentiment import get_sentiment
from dreamsApp.core.embeddings import get_text_embedding, get_image_embedding

logger = logging.getLogger(__name__)


class DreamsPipeline:
    """
    The central orchestration engine for the DREAMS algorithm.
    Simplified to focus strictly on Captioning, Sentiment, and Embeddings.

    Parameters
    ----------
    config:
        A :class:`PipelineConfig` instance controlling model IDs.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
    def process_new_post(self, user_id: str, image_path: str, caption: str, timestamp_iso: str = None) -> Dict[str, Any]:
        """
        Executes the computer vision and natural language AI ingestion sequences.
        Returns a dictionary of structured data ready for database insertion.
        """
        if timestamp_iso is None:
            timestamp_dt = datetime.now(timezone.utc)
        else:
            timestamp_dt = datetime.fromisoformat(timestamp_iso)
            
        # 1. Simple Sentiment (on Caption)
        sentiment = get_sentiment(
            caption,
            sentiment_model_name=self.config.sentiment_model_id
        )

        # 2. Caption Embedding (Text)
        caption_embedding = get_text_embedding(caption, self.config.text_embedding_model_id)

        # 3. Image Embeddings (CLIP)
        image_embedding = None
        if self.config.enable_image_embedding:
            image_embedding = get_image_embedding(image_path, self.config.image_embedding_model_id)

        post_doc = {
            'user_id': user_id,
            'caption': caption,
            'timestamp': timestamp_dt,
            'image_path': image_path,
            'sentiment': sentiment,
        }

        return {
            "post_doc": post_doc,
            "caption_embedding": caption_embedding,
            "image_embedding": image_embedding
        }

    def generate_narrative_metrics(self, user_id: str, user_posts: List[Dict]) -> Dict[str, Any]:
        from datetime import timedelta
        from dreamsApp.core.graph.builder import build_emotion_timeline
        from dreamsApp.core.graph.episode_segmentation import segment_timeline_to_episodes
        from dreamsApp.core.graph.temporal_narrative_graph import build_narrative_graph
        from dreamsApp.core.graph.graph_analysis import analyze_narrative_graph

        gap_threshold = timedelta(hours=self.config.gap_threshold_hours)
        adjacency_threshold = timedelta(days=self.config.adjacency_threshold_days)

        records = []
        for post in user_posts:
            ts = post['timestamp']
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts))

            sentiment = post.get('sentiment', {})
            records.append({
                'timestamp': ts,
                'emotion_label': sentiment.get('label', 'neutral'),
                'score': sentiment.get('score'),
                'source_id': str(post.get('_id', '')),
            })

        timeline = build_emotion_timeline(
            subject_id=user_id,
            records=records,
        )
        episodes = segment_timeline_to_episodes(
            timeline,
            gap_threshold=gap_threshold,
        )
        narrative_graph = build_narrative_graph(
            episodes,
            adjacency_threshold=adjacency_threshold,
        )
        metrics = analyze_narrative_graph(narrative_graph)
        
        return metrics
