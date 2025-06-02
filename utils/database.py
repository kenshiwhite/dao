import logging
import os
import time
from typing import Optional, List, Tuple
import psycopg2
from psycopg2 import pool, OperationalError
from contextlib import contextmanager
from dotenv import load_dotenv
import hashlib

# Загружаем переменные окружения из .env
load_dotenv()

class Database:
    def __init__(self, max_retries: int = 3, retry_delay: int = 1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialize_pool()

    def _initialize_pool(self):
        retries = 0
        while retries < self.max_retries:
            try:
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=os.getenv("DB_HOST"),
                    database=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    port=int(os.getenv("DB_PORT"))
                )
                print("✅ Успешно подключено к базе данных!")
                return
            except OperationalError as e:
                retries += 1
                safe_message = str(e).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                print(f"❌ Ошибка при подключении: {safe_message}")
                if retries >= self.max_retries:
                    raise ConnectionError("Не удалось подключиться к базе данных после нескольких попыток.")
                time.sleep(self.retry_delay)

    @contextmanager
    def get_cursor(self):
        """Контекстный менеджер для работы с курсором"""
        if not self.connection_pool:
            raise ConnectionError("Пул соединений не инициализирован")

        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"❌ Ошибка выполнения запроса: {e}")
            raise
        finally:
            self.connection_pool.putconn(conn)

    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = False):
        """Универсальный метод для выполнения запросов."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()

    def create_tables(self):
        """Создание всех необходимых таблиц"""
        # Таблица пользователей
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)

        # Таблица запросов
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                image_path TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
            )
        """)

        # Таблица классификаций
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS classifications (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                top_classes TEXT NOT NULL,
                top_probs TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)

        # Таблица избранного
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS favorites (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                image_path TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, image_path)
            )
        """)

        # Таблица отзывов
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                user_name TEXT NOT NULL,
                feedback_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def save_feedback(self, user_id: int, feedback_text: str, user_name: str = None):
        """Save user feedback to the database

        Args:
            user_id: The ID of the user submitting feedback
            feedback_text: The feedback text content
            user_name: Optional username (if not provided, will be fetched from user_id)
        """
        if user_name is None:
            # Get username from user_id if not provided
            user_name_result = self.execute_query(
                "SELECT username FROM users WHERE id = %s",
                (user_id,),
                fetch=True
            )
            if user_name_result:
                user_name = user_name_result[0][0]
            else:
                user_name = "Unknown"  # Fallback if user not found

        query = """
                INSERT INTO feedback (user_id, user_name, feedback_text)
                VALUES (%s, %s, %s)
                """
        self.execute_query(query, (user_id, user_name, feedback_text))

    def delete_feedback(self, feedback_id: int, user_id: int = None, is_admin: bool = False):
        """Delete feedback from the database

        Args:
            feedback_id: ID of the feedback to delete
            user_id: ID of the user attempting to delete (for permission check)
            is_admin: Whether the user has admin privileges

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        # If admin, delete any feedback
        if is_admin:
            self.execute_query(
                "DELETE FROM feedback WHERE id = %s",
                (feedback_id,)
            )
            return True

        # If regular user, only delete their own feedback
        if user_id:
            result = self.execute_query(
                "DELETE FROM feedback WHERE id = %s AND user_id = %s",
                (feedback_id, user_id),
                fetch=True
            )
            # Check if any rows were affected (deleted)
            return result is not None and len(result) > 0

        return False

    def authenticate_user_by_id(self, user_id: int):
        """Get user role from user_id

        Args:
            user_id: The ID of the user

        Returns:
            Tuple of (user_id, role) or (None, None) if not found
        """
        result = self.execute_query(
            "SELECT id, role FROM users WHERE id = %s",
            (user_id,),
            fetch=True
        )
        if result and len(result) > 0:
            return result[0]
        return None, None

    def get_feedbacks(self, user_id: int = None, user_name: str = None):
        """Get feedback history for a user by ID or name

        Args:
            user_id: User ID to filter by (optional)
            user_name: Username to filter by (optional)
        """
        try:
            query = """
                        SELECT u.username, f.feedback_text, f.created_at, f.id
                        FROM feedback f
                        JOIN users u ON f.user_id = u.id
                        ORDER BY f.created_at DESC
                    """

            results = self.execute_query(query, fetch=True)

            if not results:
                return []

            feedbacks = []
            for row in results:
                username, feedback_text, created_at, feedback_id = row
                feedbacks.append({
                    "username": username,
                    "feedback": feedback_text,
                    "created_at": created_at.isoformat() if created_at else None,
                    "id": feedback_id
                })

            return feedbacks

        except Exception as e:
            logging.error(f"Error fetching all feedbacks with username: {str(e)}")
            return []

    # --- Методы работы с избранным ---
    def add_to_favorites(self, user_id: int, image_path: str):
        self.execute_query(
            "INSERT INTO favorites (user_id, image_path) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (user_id, image_path)
        )

    def remove_from_favorites(self, user_id: int, image_path: str):
        self.execute_query(
            "DELETE FROM favorites WHERE user_id = %s AND image_path = %s",
            (user_id, image_path)
        )

    def get_favorites(self, user_id: int) -> List[str]:
        results = self.execute_query(
            "SELECT image_path FROM favorites WHERE user_id = %s ORDER BY added_at DESC",
            (user_id,),
            fetch=True
        )
        return [row[0] for row in results] if results else []

    # --- Методы работы с запросами ---
    def save_query(self, query_text: str, image_path: str, user_id: int):
        self.execute_query(
            "INSERT INTO queries (query_text, image_path, user_id) VALUES (%s, %s, %s)",
            (query_text, image_path, user_id)
        )

    # --- Методы работы с классификациями ---
    def save_classification(self, user_id, image_path, top_classes, top_probs, timestamp):
        query = """
            INSERT INTO classifications (user_id, image_path, top_classes, top_probs, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """
        self.execute_query(query, (user_id, image_path, ','.join(top_classes), ','.join(map(str, top_probs)), timestamp))

    def get_recent_classifications(self, user_id):
        query = """
            SELECT image_path, top_classes, top_probs, timestamp
            FROM classifications
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT 10
        """
        rows = self.execute_query(query, (user_id,), fetch=True)
        results = []
        for row in rows:
            results.append({
                "image_path": row[0],
                "top_classes": row[1].split(','),
                "top_probs": list(map(float, row[2].split(','))),
                "timestamp": row[3]
            })
        return results

    def get_top_queries(self, limit: int = 3) -> List[Tuple[str, int]]:
        query = """
            SELECT query_text, COUNT(*) AS count
            FROM queries
            GROUP BY query_text
            ORDER BY count DESC
            LIMIT %s
        """
        results = self.execute_query(query, (limit,), fetch=True)
        return results if results else []

    def get_recent_queries(self, user_id: int, limit: int = 10) -> List[str]:
        """Get recent queries for a specific user, returning only query text as a list

        Args:
            user_id: The user ID to get queries for
            limit: Maximum number of queries to return (default 10)

        Returns:
            List of query text strings
        """
        try:
            # Updated query to get the most recent queries (not distinct to preserve order)
            query = """
                SELECT query_text, timestamp
                FROM queries
                WHERE user_id = %s 
                AND query_text IS NOT NULL 
                AND query_text != ''
                AND TRIM(query_text) != ''
                ORDER BY timestamp DESC
                LIMIT %s
            """

            logging.info(f"🔍 Executing query for user_id: {user_id}, limit: {limit}")

            results = self.execute_query(query, (user_id, limit), fetch=True)

            logging.info(f"📋 Raw query results: {len(results) if results else 0} rows")

            if not results:
                logging.info("📋 No results returned from database")
                return []

            # Extract query text and remove duplicates while preserving order
            seen_queries = set()
            recent_queries = []

            for row in results:
                query_text = row[0]
                if query_text and query_text.strip():
                    cleaned_query = query_text.strip().lower()  # Normalize for duplicate checking
                    if cleaned_query not in seen_queries:
                        seen_queries.add(cleaned_query)
                        recent_queries.append(query_text.strip())  # Keep original case

            logging.info(f"📋 Processed queries: {len(recent_queries)} unique queries")
            return recent_queries[:limit]  # Ensure we don't exceed limit

        except Exception as e:
            logging.error(f"❌ Database error in get_recent_queries: {str(e)}")
            import traceback
            logging.error(f"🔍 Traceback: {traceback.format_exc()}")
            return []

    # --- Методы для пользователей ---
    def register_user(self, username: str, password: str, role: str = 'user'):
        hashed_password = self.hash_password(password)
        self.execute_query(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
            (username, hashed_password, role)
        )

    def hash_password(self, password: str):
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, username: str, password: str):
        user = self.execute_query(
            "SELECT id, password, role FROM users WHERE username = %s",
            (username,),
            fetch=True
        )
        if user:
            user_id, stored_password, role = user[0]
            if stored_password == self.hash_password(password):
                return user_id, role
        return None, None

    def check_user_role(self, user_id: int):
        role = self.execute_query(
            "SELECT role FROM users WHERE id = %s", (user_id,), fetch=True
        )
        return role[0][0] if role else 'guest'

    def access_control(self, user_id: int, required_role: str):
        user_role = self.check_user_role(user_id)
        if user_role != required_role:
            raise PermissionError("У вас нет прав для выполнения этой операции.")

    def close_all(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            print("Все соединения с базой данных закрыты.")
        else:
            print("Пул соединений не инициализирован.")
