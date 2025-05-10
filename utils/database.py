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
        """Используем контекстный менеджер для работы с курсором"""
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
        """Создание таблицы пользователей и запросов."""
        # Таблица пользователей для авторизации
        self.execute_query("""
                           CREATE TABLE IF NOT EXISTS users
                           (
                               id SERIAL PRIMARY KEY,
                               username TEXT NOT NULL UNIQUE,
                               password TEXT NOT NULL,
                               role TEXT NOT NULL DEFAULT 'user',  -- Роль может быть 'user' или 'admin'
                               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               last_login TIMESTAMP
                           )
                           """)

        # Таблица запросов
        self.execute_query("""
                           CREATE TABLE IF NOT EXISTS queries
                           (
                               id         SERIAL PRIMARY KEY,
                               query_text TEXT NOT NULL,
                               image_path TEXT NOT NULL,
                               timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL
                           )
                           """)

    def save_query(self, query_text: str, image_path: str, user_id: int):
        """Сохранение одного запроса в базу данных."""
        self.execute_query(
            "INSERT INTO queries (query_text, image_path, user_id) VALUES (%s, %s, %s)",
            (query_text, image_path, user_id)
        )

    def get_recent_queries(self, limit: int = 10) -> List[Tuple[str, str]]:
        """Получение последних запросов."""
        results = self.execute_query(
            "SELECT query_text, image_path FROM queries ORDER BY timestamp DESC LIMIT %s",
            (limit,),
            fetch=True
        )
        return results if results else []

    def register_user(self, username: str, password: str, role: str = 'user'):
        """Регистрация нового пользователя."""
        hashed_password = self.hash_password(password)
        self.execute_query(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
            (username, hashed_password, role)
        )

    def hash_password(self, password: str):
        """Хеширование пароля (используем hashlib для примера)."""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, username: str, password: str):
        """Аутентификация пользователя."""
        user = self.execute_query(
            "SELECT id, password, role FROM users WHERE username = %s",
            (username,),
            fetch=True
        )
        if user:
            user_id, stored_password, role = user[0]
            if stored_password == self.hash_password(password):
                return user_id, role
        return None, None  # Если аутентификация не удалась

    def check_user_role(self, user_id: int):
        """Проверка роли пользователя для определения прав доступа."""
        role = self.execute_query(
            "SELECT role FROM users WHERE id = %s", (user_id,), fetch=True
        )
        return role[0][0] if role else 'guest'

    def access_control(self, user_id: int, required_role: str):
        """Проверка прав доступа пользователя к ресурсу."""
        user_role = self.check_user_role(user_id)
        if user_role != required_role:
            raise PermissionError("У вас нет прав для выполнения этой операции.")

    def close_all(self):
        """Закрытие всех соединений из пула, если пул существует и не был закрыт ранее."""
        if self.connection_pool:
            try:
                # Проверка, был ли уже закрыт пул
                if not self.connection_pool.closed:
                    self.connection_pool.closeall()
                    print("✅ Все соединения закрыты.")
                else:
                    print("⚠️ Пул соединений уже закрыт.")
            except psycopg2.pool.PoolError as e:
                print(f"❌ Ошибка при закрытии пула соединений: {e}")

    def __del__(self):
        """Обеспечиваем корректное закрытие при удалении объекта."""
        self.close_all()
