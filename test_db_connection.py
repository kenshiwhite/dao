from utils.database import DatabaseManager

try:
    db = DatabaseManager()
    with db.get_cursor() as cursor:
        cursor.execute('SELECT VERSION()')
        db_version = cursor.fetchone()
        print('Database version:', db_version[0])

except Exception as e:
    print(f"Ошибка подключения: {e}")
