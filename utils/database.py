import psycopg2
from psycopg2 import sql

# Database connection parameters
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "050228Aa"  # Replace with your PostgreSQL password
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def save_query_to_db(description, image_path):
    """Save a query (description and image path) to the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = sql.SQL("""
            INSERT INTO queried_photos (description, image_path)
            VALUES (%s, %s)
        """)
        cur.execute(query, (description, image_path))
        conn.commit()
    except Exception as e:
        print(f"Error saving query to database: {e}")
    finally:
        cur.close()
        conn.close()

def get_recent_queries(limit=10):
    """Retrieve recent queries from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = sql.SQL("""
            SELECT description, image_path, query_time
            FROM queried_photos
            ORDER BY query_time DESC
            LIMIT %s
        """)
        cur.execute(query, (limit,))
        results = cur.fetchall()
        return results
    except Exception as e:
        print(f"Error retrieving recent queries: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def get_most_repeated_descriptions(limit=5):
    """Retrieve the most repeated descriptions and their photos."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = sql.SQL("""
            SELECT description, image_path, COUNT(*) as frequency
            FROM queried_photos
            GROUP BY description, image_path
            ORDER BY frequency DESC
            LIMIT %s
        """)
        cur.execute(query, (limit,))
        results = cur.fetchall()
        return results
    except Exception as e:
        print(f"Error retrieving most repeated descriptions: {e}")
        return []
    finally:
        cur.close()
        conn.close()
