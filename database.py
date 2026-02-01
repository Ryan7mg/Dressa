"""
database.py - SQLite Database Manager for Dressa App

Handles:
- User session management
- Upload tracking
- Rating storage
- Corpus growth tracking
"""

import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent / "user_study.db"


class Database:
    """SQLite database manager for user study data."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Uploads table (user's query images)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    filepath TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    added_to_corpus INTEGER DEFAULT 0,
                    num_ratings INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Ratings table (legacy, kept for backward compatibility)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    upload_id TEXT,
                    result_image_path TEXT,
                    model TEXT,
                    rating TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
                )
            """)

            # New evaluation_ratings table with provenance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_ratings (
                    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    upload_id TEXT NOT NULL,
                    result_image_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    display_position INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ==================== User Methods ====================

    def create_user(self) -> str:
        """
        Create a new user with a unique ID.

        Returns:
            user_id: UUID string for the new user
        """
        user_id = str(uuid.uuid4())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (user_id) VALUES (?)",
                (user_id,)
            )
            conn.commit()

        logger.info(f"Created new user: {user_id}")
        return user_id

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def user_exists(self, user_id: str) -> bool:
        """Check if a user exists."""
        return self.get_user(user_id) is not None

    # ==================== Upload Methods ====================

    def create_upload(self, user_id: str, filepath: str) -> str:
        """
        Record a new image upload.

        Args:
            user_id: ID of the user who uploaded
            filepath: Path where the uploaded image is stored

        Returns:
            upload_id: UUID string for the new upload
        """
        upload_id = str(uuid.uuid4())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO uploads (upload_id, user_id, filepath)
                   VALUES (?, ?, ?)""",
                (upload_id, user_id, filepath)
            )
            conn.commit()

        logger.info(f"Created upload {upload_id} for user {user_id}")
        return upload_id

    def get_upload(self, upload_id: str) -> Optional[Dict]:
        """Get upload by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM uploads WHERE upload_id = ?",
                (upload_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_uploads(self, user_id: str) -> List[Dict]:
        """Get all uploads for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM uploads
                   WHERE user_id = ?
                   ORDER BY uploaded_at DESC""",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def mark_added_to_corpus(self, upload_id: str):
        """Mark an upload as added to the corpus."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE uploads
                   SET added_to_corpus = 1
                   WHERE upload_id = ?""",
                (upload_id,)
            )
            conn.commit()

        logger.info(f"Marked upload {upload_id} as added to corpus")

    def increment_upload_ratings(self, upload_id: str) -> int:
        """
        Increment the rating count for an upload.

        Returns:
            New rating count
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE uploads
                   SET num_ratings = num_ratings + 1
                   WHERE upload_id = ?""",
                (upload_id,)
            )
            cursor.execute(
                "SELECT num_ratings FROM uploads WHERE upload_id = ?",
                (upload_id,)
            )
            conn.commit()
            result = cursor.fetchone()
            return result['num_ratings'] if result else 0

    # ==================== Rating Methods ====================

    def save_rating(
        self,
        user_id: str,
        upload_id: str,
        result_image_path: str,
        model: str,
        rating: str
    ) -> int:
        """
        Save a user's rating for a result image.

        Args:
            user_id: ID of the user
            upload_id: ID of the query image upload
            result_image_path: Path to the result image being rated
            model: Which model returned this result
            rating: 'similar' or 'not_similar'

        Returns:
            rating_id: ID of the new rating
        """
        if rating not in ('similar', 'not_similar'):
            raise ValueError(f"Invalid rating: {rating}. "
                           f"Must be 'similar' or 'not_similar'")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO ratings
                   (user_id, upload_id, result_image_path, model, rating)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, upload_id, result_image_path, model, rating)
            )
            conn.commit()
            rating_id = cursor.lastrowid

        # Increment upload's rating count
        self.increment_upload_ratings(upload_id)

        logger.info(f"Saved rating {rating_id}: {rating} for {model}")
        return rating_id

    def get_upload_ratings(self, upload_id: str) -> List[Dict]:
        """Get all ratings for an upload."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM ratings
                   WHERE upload_id = ?
                   ORDER BY timestamp""",
                (upload_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_user_ratings(self, user_id: str) -> List[Dict]:
        """Get all ratings by a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM ratings
                   WHERE user_id = ?
                   ORDER BY timestamp""",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def has_rated(self, upload_id: str, result_image_path: str) -> bool:
        """Check if a result image has already been rated for this upload."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT COUNT(*) as count FROM ratings
                   WHERE upload_id = ? AND result_image_path = ?""",
                (upload_id, result_image_path)
            )
            result = cursor.fetchone()
            return result['count'] > 0

    # ==================== Evaluation Rating Methods ====================

    def save_evaluation_rating(
        self,
        user_id: str,
        upload_id: str,
        result_image_id: str,
        rating: str,
        provenance: dict,
        display_position: int
    ):
        """
        Save a user's evaluation rating with provenance information.

        Args:
            user_id: ID of the user
            upload_id: ID of the query image upload
            result_image_id: Path/ID of the result image being rated
            rating: 'similar' or 'not_similar'
            provenance: Dict mapping model_name -> rank (1-indexed)
            display_position: Position in the shuffled display order
        """
        if rating not in ('similar', 'not_similar'):
            raise ValueError(f"Invalid rating: {rating}. "
                           f"Must be 'similar' or 'not_similar'")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluation_ratings
                (user_id, upload_id, result_image_id, rating, provenance, display_position)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, upload_id, result_image_id, rating,
                  json.dumps(provenance), display_position))
            conn.commit()

        # Increment upload's rating count
        self.increment_upload_ratings(upload_id)

        logger.info(f"Saved evaluation rating: {rating} for {result_image_id} "
                   f"(position {display_position}, provenance: {provenance})")

    def get_evaluation_ratings(self, upload_id: str) -> List[Dict]:
        """Get all evaluation ratings for an upload."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM evaluation_ratings
                   WHERE upload_id = ?
                   ORDER BY timestamp""",
                (upload_id,)
            )
            rows = [dict(row) for row in cursor.fetchall()]
            # Parse provenance JSON
            for row in rows:
                row['provenance'] = json.loads(row['provenance'])
            return rows

    # ==================== Analytics Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total users
            cursor.execute("SELECT COUNT(*) as count FROM users")
            stats['total_users'] = cursor.fetchone()['count']

            # Total uploads
            cursor.execute("SELECT COUNT(*) as count FROM uploads")
            stats['total_uploads'] = cursor.fetchone()['count']

            # Uploads added to corpus
            cursor.execute(
                "SELECT COUNT(*) as count FROM uploads WHERE added_to_corpus = 1"
            )
            stats['corpus_additions'] = cursor.fetchone()['count']

            # Total ratings
            cursor.execute("SELECT COUNT(*) as count FROM ratings")
            stats['total_ratings'] = cursor.fetchone()['count']

            # Ratings by model
            cursor.execute(
                """SELECT model, COUNT(*) as count
                   FROM ratings GROUP BY model"""
            )
            stats['ratings_by_model'] = {
                row['model']: row['count']
                for row in cursor.fetchall()
            }

            # Similar vs not similar
            cursor.execute(
                """SELECT rating, COUNT(*) as count
                   FROM ratings GROUP BY rating"""
            )
            stats['ratings_breakdown'] = {
                row['rating']: row['count']
                for row in cursor.fetchall()
            }

            return stats

    def export_ratings_csv(self, output_path: str):
        """Export all ratings to CSV for analysis."""
        import csv

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, u.filepath as query_image_path
                FROM ratings r
                JOIN uploads u ON r.upload_id = u.upload_id
                ORDER BY r.timestamp
            """)
            rows = cursor.fetchall()

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rating_id', 'user_id', 'upload_id', 'query_image_path',
                'result_image_path', 'model', 'rating', 'timestamp'
            ])
            for row in rows:
                writer.writerow([
                    row['rating_id'], row['user_id'], row['upload_id'],
                    row['query_image_path'], row['result_image_path'],
                    row['model'], row['rating'], row['timestamp']
                ])

        logger.info(f"Exported {len(rows)} ratings to {output_path}")


# Convenience function for testing
def test_database():
    """Test database operations."""
    import tempfile
    import os

    # Use temp database for testing
    test_db = Path(tempfile.gettempdir()) / "dressa_test.db"
    if test_db.exists():
        os.remove(test_db)

    db = Database(test_db)

    print("Testing database.py...")

    # Test user creation
    user_id = db.create_user()
    print(f"  Created user: {user_id[:8]}...")

    # Test upload creation
    upload_id = db.create_upload(user_id, "/path/to/test.jpg")
    print(f"  Created upload: {upload_id[:8]}...")

    # Test rating
    rating_id = db.save_rating(
        user_id, upload_id, "/path/to/result.jpg",
        "openai_clip", "similar"
    )
    print(f"  Saved rating: {rating_id}")

    # Test stats
    stats = db.get_stats()
    print(f"  Stats: {stats}")

    # Cleanup
    os.remove(test_db)
    print("\nDatabase tests complete!")


if __name__ == "__main__":
    test_database()
