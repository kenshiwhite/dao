import sqlite3

def create_db():
    conn = sqlite3.connect('queries.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,       -- Используем INTEGER PRIMARY KEY для автоинкремента
            query_text TEXT NOT NULL,
            result_image_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP    
        )
    ''')

    conn.commit()
    conn.close()

def insert_query(query_text, image_path):
    conn = sqlite3.connect('queries.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO queries (query_text, result_image_path)
        VALUES (?, ?)
    ''', (query_text, image_path))

    conn.commit()
    conn.close()

# Получение самых популярных запросов
def get_most_popular_queries(limit=10):
    conn = sqlite3.connect('queries.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT query_text, COUNT(*) as count
        FROM queries
        GROUP BY query_text
        ORDER BY count DESC
        LIMIT ?
    ''', (limit,))

    popular_queries = cursor.fetchall()
    conn.close()
    return popular_queries

create_db()  # Создаем базу данных и таблицу


popular_queries = get_most_popular_queries()
for query, count in popular_queries:
    print(f"Запрос: {query}, Количество: {count}")
