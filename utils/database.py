import os
import psycopg2
from psycopg2 import pool, OperationalError
from contextlib import contextmanager
import time
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class database:
    def __init__(self, max_retries=3, retry_delay=1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_pool = None
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

    def execute_query(self, query, params=None, fetch=False):
        """Универсальный метод для выполнения запросов."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()

    def create_tables(self):
        """Создание таблиц."""
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                image_path TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def save_query(self, query_text, image_path):
        """Сохранение запроса в базу данных."""
        self.execute_query(
            "INSERT INTO queries (query_text, image_path) VALUES (%s, %s)",
            (query_text, image_path)
        )

    def get_recent_queries(self, limit=10):
        """Получение последних запросов."""
        return self.execute_query(
            "SELECT query_text, image_path FROM queries ORDER BY timestamp DESC LIMIT %s",
            (limit,),
            fetch=True
        )

    def close_all(self):
        """Закрыть все соединения."""
        if self.connection_pool:
            self.connection_pool.closeall()

    def __del__(self):
        self.close_all()
