"""Vector store utilities for managing Multiplex Graph embeddings."""

import os
import hashlib
import logging
from typing import List, Dict, Any, Optional

try:
    import chromadb
except ImportError:
    logging.warning("chromadb not installed. Vector storage will be disabled.")
    chromadb = None

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages persistent ChromaDB collections for Multiplex Graph layers."""
    
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.client = None
        self.collections = {}
        
        if chromadb is None:
            return
            
        try:
            # Ensure directory exists
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Initialize Multiplex Layer collections
            self.collections = {
                "layer_2_semantic": self.client.get_or_create_collection(
                    name="multiplex_layer_2_semantic",
                    metadata={"description": "OSM taxonomy embeddings"}
                ),
                "layer_3_visual": self.client.get_or_create_collection(
                    name="multiplex_layer_3_visual",
                    metadata={"description": "CLIP image embeddings"}
                ),
                "layer_5_emotional": self.client.get_or_create_collection(
                    name="multiplex_layer_5_emotional",
                    metadata={"description": "DistilRoBERTa emotion embeddings"}
                ),
                # General collection for the extracted keywords (backward compatibility)
                "keywords": self.client.get_or_create_collection(
                    name="extracted_keywords",
                    metadata={"description": "Standalone keyword embeddings"}
                )
            }
            logger.info(f"Initialized ChromaDB at {persist_dir} with {len(self.collections)} collections")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None

    def store_vector(self, collection_name: str, doc_id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """Store a single embedding vector in the specified collection."""
        if not self.client or collection_name not in self.collections:
            return False
            
        try:
            self.collections[collection_name].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store vector in {collection_name}: {e}")
            return False
            
    def store_keywords(self, user_id: str, post_id: str, keywords_data: List[Dict[str, Any]]):
        """Store a batch of keywords into the keywords collection.
        
        Args:
            user_id: The owner of the keywords
            post_id: The post they belong to
            keywords_data: List of dicts containing 'keyword' and 'embedding'
        """
        if not self.client or "keywords" not in self.collections or not keywords_data:
            return False
            
        try:
            ids = [
                f"{post_id}_{hashlib.sha256(k['keyword'].lower().strip().encode()).hexdigest()[:12]}"
                for k in keywords_data
            ]
            embeddings = [k["embedding"] for k in keywords_data]
            metadatas = [{"user_id": user_id, "post_id": post_id, "keyword": k["keyword"]} for k in keywords_data]
            
            self.collections["keywords"].upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store keywords batch: {e}")
            return False


# Singleton instance
vector_store = VectorStore()
