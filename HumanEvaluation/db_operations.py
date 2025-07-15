# db_operations.py
import sqlite3
import os
import random
import pandas as pd # Import pandas
from datetime import datetime

DATABASE_NAME = 'evaluations.db'

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    return conn

def create_tables():
    """Creates all necessary tables if they don't already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            question_id INTEGER PRIMARY KEY,
            question_text TEXT NOT NULL
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS approaches (
            approach_id INTEGER PRIMARY KEY AUTOINCREMENT,
            approach_name TEXT NOT NULL UNIQUE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            approach_id INTEGER,
            response_text TEXT NOT NULL,
            FOREIGN KEY (question_id) REFERENCES questions (question_id),
            FOREIGN KEY (approach_id) REFERENCES approaches (approach_id)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question_id INTEGER,
            response_id INTEGER,
            relevance_rating INTEGER NOT NULL,
            coherence_rating INTEGER NOT NULL,
            completeness_rating INTEGER NOT NULL,
            conciseness_rating INTEGER NOT NULL,
            notes TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (question_id) REFERENCES questions (question_id),
            FOREIGN KEY (response_id) REFERENCES responses (response_id)
        );
    ''')
    conn.commit()
    conn.close()

def ingest_dataframe_data(df: pd.DataFrame):
    """
    Ingests data from a Pandas DataFrame into the questions, approaches, and responses tables.
    DataFrame is expected to have 'qid' (question text) and 'ap1' through 'ap5' (response texts).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Ensure approaches are populated and get their IDs
    approach_names = ['zero-shot', 'rag', 'finetune', 'draft-flant5', 'draft-llama']
    approach_ids = {}
    for name in approach_names:
        cursor.execute("INSERT INTO approaches (approach_name) VALUES (?)", (name,))
        conn.commit() # Commit after each insert to ensure ID is available
        cursor.execute("SELECT approach_id FROM approaches WHERE approach_name = ?", (name,))
        approach_ids[name] = cursor.fetchone()[0]
    print("Approaches ensured:", approach_ids)

    # Ingest questions and responses from DataFrame
    for index, row in df.iterrows():
        question_id = row['id'] # Assuming 'qid' column contains the question text
        question_text = row['Context'] # Assuming 'qid' column contains the question text

        # Insert question if it doesn't exist
        cursor.execute("INSERT INTO questions (question_text, question_id) VALUES (?, ?)", (question_text, question_id))
        conn.commit() # Commit to ensure lastrowid is correct

        # Insert responses for this question
        for i in range(5): # For ap1 to ap5
            approach_name = approach_names[i]
            response_text = row[approach_name]
            approach_id = approach_ids[approach_name]

            # Check if response already exists for this question and approach
            cursor.execute('''
                SELECT response_id FROM responses
                WHERE question_id = ? AND approach_id = ?
            ''', (question_id, approach_id))
            existing_response = cursor.fetchone()

            if not existing_response:
                cursor.execute(
                    "INSERT INTO responses (question_id, approach_id, response_text) VALUES (?, ?, ?)",
                    (question_id, approach_id, response_text)
                )
    conn.commit()
    conn.close()
    print("DataFrame data ingested successfully.")

def add_sample_data():
    """
    Generates a dummy DataFrame and calls ingest_dataframe_data to populate the DB.
    This function is for initial setup/testing. In a real scenario, the user
    would load their DataFrame and call ingest_dataframe_data directly.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if questions already exist to avoid re-ingestion of sample data
    cursor.execute("SELECT COUNT(*) FROM questions")
    if cursor.fetchone()[0] > 0:
        conn.close()
        print("Database already contains questions. Skipping sample data generation.")
        return

    print("Generating and ingesting sample DataFrame data...")

    dummy_df = pd.read_feather("../data")
    ingest_dataframe_data(dummy_df)
    conn.close() # Close connection opened by add_sample_data if it didn't return early


def get_user_by_email(email):
    """Retrieves a user by email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(name, email):
    """Creates a new user and returns their user_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None # Email already exists
    finally:
        conn.close()

def get_next_unrated_question(user_id):
    """
    Finds the next question that the user has not fully rated yet.
    A question is fully rated if there are 5 ratings entries for it by the user.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT q.question_id, q.question_text
        FROM questions q
        LEFT JOIN (
            SELECT question_id, COUNT(DISTINCT response_id) as rated_responses_count
            FROM ratings
            WHERE user_id = ?
            GROUP BY question_id
        ) r ON q.question_id = r.question_id
        WHERE r.rated_responses_count IS NULL OR r.rated_responses_count < 5
        ORDER BY q.question_id ASC
        LIMIT 1;
    ''', (user_id,))
    question = cursor.fetchone()
    conn.close()
    return question

def get_responses_for_question(question_id):
    """Retrieves all responses for a given question_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT response_id, response_text, approach_id FROM responses WHERE question_id = ?",
        (question_id,)
    )
    responses = cursor.fetchall()
    conn.close()
    return responses

def save_rating(user_id, question_id, response_id, relevance, coherence, completeness, conciseness, notes):
    """Saves a single rating for a response."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()

    # Check if a rating for this user, question, and response already exists
    cursor.execute('''
        SELECT rating_id FROM ratings
        WHERE user_id = ? AND question_id = ? AND response_id = ?
    ''', (user_id, question_id, response_id))
    existing_rating = cursor.fetchone()

    if existing_rating:
        # Update existing rating
        cursor.execute('''
            UPDATE ratings
            SET relevance_rating = ?, coherence_rating = ?, completeness_rating = ?, conciseness_rating = ?, notes = ?, timestamp = ?
            WHERE rating_id = ?
        ''', (relevance, coherence, completeness, conciseness, notes, timestamp, existing_rating['rating_id']))
    else:
        # Insert new rating
        cursor.execute('''
            INSERT INTO ratings (user_id, question_id, response_id, relevance_rating, coherence_rating, completeness_rating, conciseness_rating, notes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, question_id, response_id, relevance, coherence, completeness, conciseness, notes, timestamp))
    conn.commit()
    conn.close()


# Initialize DB and add sample data when the script is imported/run for the first time
create_tables()
add_sample_data() # This will now call ingest_dataframe_data with a dummy DataFrame
