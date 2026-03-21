import sqlite3
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLiteManager:
    """Manages the structured metadata mapping to SQL arrays."""
    def __init__(self, db_path: str = "./data/dreams.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        image_path TEXT,
                        caption TEXT,
                        timestamp TEXT,
                        sentiment_label TEXT,
                        sentiment_score REAL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        email TEXT UNIQUE,
                        password_hash TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS thematic_analysis (
                        user_id TEXT PRIMARY KEY,
                        data_json TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS keywords (
                        user_id TEXT PRIMARY KEY,
                        positive_keywords_json TEXT,
                        negative_keywords_json TEXT,
                        clustered_keywords_json TEXT
                    )
                """)
                
                # Attempt to add new columns to existing posts table if it was created before this update
                try:
                    cursor.execute("ALTER TABLE posts ADD COLUMN corrected_label TEXT")
                except sqlite3.OperationalError:
                    pass
                    
                try:
                    cursor.execute("ALTER TABLE posts ADD COLUMN chime_analysis_json TEXT")
                except sqlite3.OperationalError:
                    pass
                    
                conn.commit()
                logger.info(f"Initialized SQLite database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DB: {e}")

    def insert_post(self, user_id: str, image_path: str, caption: str, timestamp: datetime, sentiment_label: str, sentiment_score: float) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO posts (user_id, image_path, caption, timestamp, sentiment_label, sentiment_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, image_path, caption, timestamp.isoformat(), sentiment_label, sentiment_score))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert post: {e}")
            return -1

# Singleton instance
db_manager = SQLiteManager()
