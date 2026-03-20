import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from dreamsApp.core.graph.builder import build_emotion_timeline
from dreamsApp.core.graph.episode_segmentation import segment_timeline_to_episodes
from dreamsApp.core.graph.temporal_narrative_graph import build_narrative_graph
from dreamsApp.core.graph.graph_analysis import analyze_narrative_graph

from dreamsApp.core.keywords import extract_keywords_and_vectors
from dreamsApp.core.location_extractor import extract_gps_from_image
from dreamsApp.core.sentiment import (
    get_chime_category,
    get_image_caption_and_sentiment,
    select_text_for_analysis,
)

logger = logging.getLogger(__name__)

class DreamsPipeline:
    """
    The central orchestration engine for the DREAMS algorithm.
    Decoupled entirely from Flask web routes.
    """
    
    def __init__(self):
        # The pipeline can eventually hold loaded AI models directly
        # rather than reloading them in individual scripts.
        pass
        
    def process_new_post(self, user_id: str, image_path: str, caption: str, timestamp_iso: str = None) -> Dict[str, Any]:
        """
        Executes the computer vision and natural language AI ingestion sequences.
        Returns a dictionary of structured data ready for database insertion.
        """
        if timestamp_iso is None:
            timestamp_dt = datetime.now(timezone.utc)
        else:
            timestamp_dt = datetime.fromisoformat(timestamp_iso)
            
        gps_data = extract_gps_from_image(image_path)
        
        analysis_result = get_image_caption_and_sentiment(image_path, caption)
        sentiment = analysis_result["sentiment"]
        generated_caption = analysis_result["imgcaption"]

        text_for_analysis = select_text_for_analysis(caption, generated_caption)
        chime_result = get_chime_category(text_for_analysis)
        
        keywords_with_vectors = []
        keyword_type = None
        if sentiment['label'] in ('positive', 'negative'):
            keywords_with_vectors = extract_keywords_and_vectors(generated_caption)
            keyword_type = f"{sentiment['label']}_keywords"

        keywords_for_mongo = []
        if keywords_with_vectors:
            keywords_for_mongo = [
                {k: v for k, v in kw.items() if k != "embedding"}
                for kw in keywords_with_vectors
            ]

        post_doc = {
            'user_id': user_id,
            'caption': caption,
            'timestamp': timestamp_dt,
            'image_path': image_path,
            'generated_caption': generated_caption,
            'sentiment': sentiment,
            'chime_analysis': chime_result,
            'location': gps_data,
        }

        return {
            "post_doc": post_doc,
            "keyword_type": keyword_type,
            "keywords_for_db": keywords_for_mongo,
            "keywords_with_vectors": keywords_with_vectors
        }

    def generate_narrative_metrics(self, user_id: str, user_posts: List[Dict], gap_threshold_hours: int = 24, adjacency_threshold_days: int = 7) -> Dict[str, Any]:
        """
        Executes the temporal graph building sequence and calculates structural narrative metrics.
        Returns a dictionary containing the graph analytics results.
        """
        gap_threshold = timedelta(hours=gap_threshold_hours)
        adjacency_threshold = timedelta(days=adjacency_threshold_days)

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
