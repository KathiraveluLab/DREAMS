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

    def ingest_csv(self, csv_path: str) -> None:
        """
        Ingests a CSV dataset, processes it through the ML pipeline, and stores 
        results synchronously into SQLite and ChromaDB.
        """
        import pandas as pd
        import os
        from tqdm import tqdm
        from dreamsApp.core.database import db_manager
        from dreamsApp.core.vector_store import vector_store

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return

        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)

        required_cols = {"user_id", "image_path", "caption"}
        if not required_cols.issubset(df.columns):
            logger.error(f"CSV missing required columns. Expected at least: {required_cols}")
            return

        success_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting Posts"):
            user_id = str(row["user_id"])
            image_path = str(row["image_path"])
            caption = str(row["caption"])
            
            timestamp_iso = None
            if "timestamp" in df.columns and pd.notna(row["timestamp"]):
                timestamp_iso = str(row["timestamp"])
            
            try:
                result = self.process_new_post(
                    user_id=user_id,
                    image_path=image_path,
                    caption=caption,
                    timestamp_iso=timestamp_iso
                )
            except Exception as e:
                logger.error(f"Pipeline error on row {idx}: {e}")
                continue

            post_doc = result["post_doc"]
            sentiment = post_doc.get("sentiment", {})
            
            # Store metadata in SQLite
            post_id = db_manager.insert_post(
                user_id=user_id,
                image_path=image_path,
                caption=caption,
                timestamp=post_doc["timestamp"],
                sentiment_label=sentiment.get("label", "neutral"),
                sentiment_score=sentiment.get("score", 0.0)
            )

            if post_id == -1:
                logger.error(f"Failed to insert row {idx} into SQLite. Skipping vector storage.")
                continue

            # Store embeddings in ChromaDB using SQLite's ID
            doc_id = str(post_id)
            
            if result.get("caption_embedding"):
                vector_store.store_vector(
                    collection_name="text_embeddings",
                    doc_id=doc_id,
                    embedding=result["caption_embedding"],
                    metadata={"user_id": user_id, "type": "caption"}
                )
                
            if result.get("image_embedding"):
                vector_store.store_vector(
                    collection_name="image_embeddings",
                    doc_id=doc_id,
                    embedding=result["image_embedding"],
                    metadata={"user_id": user_id, "type": "image"}
                )
                
            success_count += 1

        logger.info(f"Ingestion complete. Successfully processed {success_count}/{len(df)} rows.")

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Ingest CSV dataset using DREAMS pipeline.")
    parser.add_argument("csv_path", help="Path to the CSV file.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to the override YAML config.")
    args = parser.parse_args()

    if args.config_path and os.path.exists(args.config_path):
        config = PipelineConfig.from_yaml(args.config_path)
    else:
        config = PipelineConfig()
        
    pipeline = DreamsPipeline(config=config)
    pipeline.ingest_csv(args.csv_path)
