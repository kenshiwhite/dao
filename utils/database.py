import os
import time
import hashlib
from typing import Optional, List, Tuple
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, OperationalError
from dotenv import load_dotenv

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
                print(f"❌ Ошибка при подключении: {e}")
                if retries >= self.max_retries:
                    raise ConnectionError("Не удалось подключиться к базе данных после нескольких попыток.")
                time.sleep(self.retry_delay)

    @contextmanager
    def get_cursor(self):
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
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()

    def create_tables(self):
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
        """)

        self.execute_query("""
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                image_path TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
            );
        """)

        self.execute_query("""
            CREATE TABLE IF NOT EXISTS classifications (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                top_classes TEXT NOT NULL,
                top_probs TEXT NOT NULL,
                timestamp BIGINT NOT NULL
            );
        """)

        self.execute_query("""
            CREATE TABLE IF NOT EXISTS favorites (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                image_path TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, image_path)
            );
        """)

        self.execute_query("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                query_text TEXT NOT NULL,
                uploaded_image_path TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    def log_search_query(self, user_id: Optional[int], query_text: str, uploaded_image_path: str):
        """Функция для записи запроса в таблицу search_logs"""
        query = '''INSERT INTO search_logs (user_id, query_text, uploaded_image_path)
                   VALUES (%s, %s, %s)'''
        self.execute_query(query, (user_id, query_text, uploaded_image_path))

    def get_search_logs(self, user_id: Optional[int] = None, limit: int = 10) -> List[Tuple]:
        """Функция для получения логов запросов"""
        if user_id:
            query = """
                SELECT query_text, uploaded_image_path, timestamp
                FROM search_logs
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            params = (user_id, limit)
        else:
            query = """
                SELECT query_text, uploaded_image_path, timestamp
                FROM search_logs
                ORDER BY timestamp DESC
                LIMIT %s
            """
            params = (limit,)
        return self.execute_query(query, params, fetch=True)

    def save_query(self, query_text: str, image_path: str, user_id: int):
        self.execute_query(
            "INSERT INTO queries (query_text, image_path, user_id) VALUES (%s, %s, %s)",
            (query_text, image_path, user_id)
        )

    def save_classification(self, user_id, image_path, top_classes, top_probs, timestamp):
        self.execute_query(
            "INSERT INTO classifications (user_id, image_path, top_classes, top_probs, timestamp) VALUES (%s, %s, %s, %s, %s)",
            (user_id, image_path, ','.join(top_classes), ','.join(map(str, top_probs)), timestamp)
        )

    def get_recent_classifications(self, user_id):
        results = self.execute_query(
            "SELECT image_path, top_classes, top_probs, timestamp FROM classifications WHERE user_id = %s ORDER BY timestamp DESC LIMIT 10",
            (user_id,),
            fetch=True
        )
        classifications = []
        for row in results:
            classifications.append({
                "image_path": row[0],
                "top_classes": row[1].split(','),
                "top_probs": list(map(float, row[2].split(','))),
                "timestamp": row[3]
            })
        return classifications

    def get_recent_queries(self, limit: int = 10, user_id: Optional[int] = None) -> List[Tuple]:
        if user_id:
            query = """
                SELECT q.query_text, q.image_path, u.username, q.timestamp
                FROM queries q
                LEFT JOIN users u ON q.user_id = u.id
                WHERE q.user_id = %s
                ORDER BY q.timestamp DESC
                LIMIT %s
            """
            params = (user_id, limit)
        else:
            query = """
                SELECT q.query_text, q.image_path, u.username, q.timestamp
                FROM queries q
                LEFT JOIN users u ON q.user_id = u.id
                ORDER BY q.timestamp DESC
                LIMIT %s
            """
            params = (limit,)
        return self.execute_query(query, params, fetch=True)

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

    def register_user(self, username: str, password: str, role: str = 'user'):
        hashed_password = self.hash_password(password)
        self.execute_query(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
            (username, hashed_password, role)
        )

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, username: str, password: str) -> Tuple[Optional[int], Optional[str]]:
        result = self.execute_query(
            "SELECT id, password, role FROM users WHERE username = %s",
            (username,),
            fetch=True
        )
        if result:
            user_id, stored_hash, role = result[0]
            if stored_hash == self.hash_password(password):
                return user_id, role
        return None, None

    def check_user_role(self, user_id: int) -> str:
        result = self.execute_query(
            "SELECT role FROM users WHERE id = %s",
            (user_id,),
            fetch=True
        )
        return result[0][0] if result else 'guest'

    def access_control(self, user_id: int, required_role: str):
        role = self.check_user_role(user_id)
        if role != required_role:
            raise PermissionError("У вас нет прав для выполнения этой операции.")

    def close_all(self):
        if self.connection_pool:
            try:
                self.connection_pool.closeall()
                print("✅ Все соединения закрыты.")
            except Exception as e:
                print(f"❌ Ошибка при закрытии соединений: {e}")

    def __del__(self):
        self.close_all()
