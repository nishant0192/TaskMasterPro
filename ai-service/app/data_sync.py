# ai-service/app/data_sync.py

import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

load_dotenv()

# Connection URLs from environment variables
MAIN_DB_URL = os.getenv("DATABASE_URL")       # Main TaskMasterPro DB
AI_DB_URL = os.getenv("AI_DATABASE_URL")        # Separate AI DB

def fetch_tasks_from_main_db():
    """
    Connects to the main database and fetches task data.
    Adjust the SELECT fields as necessary for your training/AI requirements.
    """
    conn = psycopg2.connect(MAIN_DB_URL)
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Fetch non-archived tasks; adjust query if more fields are needed.
            cur.execute("""
                SELECT 
                    id,
                    title,
                    description,
                    priority,
                    "dueDate",
                    "createdAt"
                FROM "Task"
                WHERE "isArchived" = false
            """)
            tasks = cur.fetchall()
    finally:
        conn.close()
    return tasks

def store_training_data_in_ai_db(tasks):
    """
    Inserts or updates task data in the AI database.
    This example assumes an AI-specific table 'training_data' to store
    essential fields for model training or prediction.
    """
    conn = psycopg2.connect(AI_DB_URL)
    try:
        with conn.cursor() as cur:
            # Create the training_data table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    task_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    priority INTEGER,
                    due_date TIMESTAMP,
                    created_at TIMESTAMP
                )
            """)
            # Insert or update each task record
            for task in tasks:
                cur.execute("""
                    INSERT INTO training_data (task_id, title, description, priority, due_date, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO UPDATE 
                    SET title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        priority = EXCLUDED.priority,
                        due_date = EXCLUDED.due_date,
                        created_at = EXCLUDED.created_at;
                """, (
                    task["id"],
                    task["title"],
                    task["description"],
                    task["priority"],
                    task["dueDate"],
                    task["createdAt"]
                ))
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def sync_tasks_to_ai_db():
    """
    Fetch tasks from the main DB and sync them to the AI database.
    """
    tasks = fetch_tasks_from_main_db()
    print(f"Fetched {len(tasks)} tasks from main DB.")
    store_training_data_in_ai_db(tasks)
    print("Training data successfully synced to the AI DB.")

if __name__ == "__main__":
    sync_tasks_to_ai_db()
