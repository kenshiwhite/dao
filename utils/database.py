import os
import psycopg2
from psycopg2 import pool, OperationalError
from datetime import datetime
from PIL import Image
from contextlib import contextmanager
import time


class DatabaseManager:
    def __init__(self, max_retries=3, retry_delay=1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_pool = None
        self._initialize_pool()
        self.create_tables()

    def _initialize_pool(self):
        retries = 0
        while retries < self.max_retries:
            try:
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=os.getenv('DB_HOST', 'localhost'),
                    database=os.getenv('DB_NAME', 'clip_search'),
                    user=os.getenv('DB_USER', 'postgres'),
                    password=os.getenv('DB_PASSWORD', '050228'),
                    port=os.getenv('DB_PORT', '5432')
                )
                return
            except OperationalError as e:
                retries += 1
                if retries >= self.max_retries:
                    raise ConnectionError(f"Failed to connect to database after {self.max_retries} attempts: {e}")
                time.sleep(self.retry_delay)

    @contextmanager
    def get_cursor(self):
        if not self.connection_pool:
            raise ConnectionError("Database connection pool not initialized")

        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.connection_pool.putconn(conn)

    def create_tables(self):
        with self.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT,
                    image_path TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_query(self, query_text, image):
        """Save a query and its result image to the database"""
        os.makedirs("data/query_results", exist_ok=True)
        image_path = f"data/query_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(image_path)

        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO queries (query_text, image_path) VALUES (%s, %s)",
                (query_text, image_path)
            )

    def get_recent_queries(self, limit=10):
        """Get recent queries from the database"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT query_text, image_path FROM queries ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
            return cursor.fetchall()

    def close_all(self):
        """Close all connections in the pool"""
        if hasattr(self, 'connection_pool') and self.connection_pool:
            self.connection_pool.closeall()

    def __del__(self):
        self.close_all()