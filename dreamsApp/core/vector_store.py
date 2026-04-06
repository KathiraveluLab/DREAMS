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
            
            # Initialize Pipeline collections
            self.collections = {
                "text_embeddings": self.client.get_or_create_collection(
                    name="pipeline_text_embeddings",
                    metadata={"description": "Text caption embeddings"}
                ),
                "image_embeddings": self.client.get_or_create_collection(
                    name="pipeline_image_embeddings",
                    metadata={"description": "CLIP vision embeddings"}
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
            



# Singleton instance
vector_store = VectorStore()
