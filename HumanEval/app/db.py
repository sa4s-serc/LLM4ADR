import pandas as pd
import psycopg2 as pg
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()


class DBConnection:
    def __init__(self, dbname, user, password, host):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = pg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host
        )
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()


class User(BaseModel):
    uid: int = None
    name: str
    email: str
    age: int
    done_se: str
    industry_exp: int
    industry_org_name: str
    rate_yourself: int
    created_at: datetime


class ExperimentRun(BaseModel):
    user_id: int
    timestamp: datetime
    context: str
    decision1: str
    decision1_rating: int
    decision1_note: str
    decision1_approach: int
    decision2: str
    decision2_rating: int
    decision2_note: str
    decision2_approach: int


class PromptLibrary(BaseModel):
    class Context(BaseModel):
        context: str
        context_id: int


class Database:
    def __init__(self):
        self.connection = DBConnection(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST")
        )

    def start(self):
        self.create_tables()

    def create_tables(self):
        with self.connection as cursor:
            # Create a users table where the id starts from 1000
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    uid SERIAL PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    age INTEGER,
                    done_se TEXT,
                    industry_exp INTEGER,
                    industry_org_name TEXT,
                    rate_yourself INTEGER,
                    created_at TIMESTAMP
                );
                """
            )

            # Set the starting value for the uid sequence only if it is less than 1000
            cursor.execute(
                """
                DO $$
                BEGIN
                    IF (SELECT last_value FROM users_uid_seq) < 1000 THEN
                        PERFORM setval('users_uid_seq', 1000, false);
                    END IF;
                END $$;
                """
            )

            # Create a table to store the experiment runs
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id SERIAL PRIMARY KEY,
                    context TEXT,
                    user_id INTEGER,
                    timestamp TIMESTAMP,
                    decision1 TEXT,
                    decision1_rating INTEGER,
                    decision1_note TEXT,
                    decision1_approach INTEGER,
                    decision2 TEXT,
                    decision2_rating INTEGER,
                    decision2_note TEXT,
                    decision2_approach INTEGER
                );
                """
            )

            # Create a table to store the custom decisions
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS custom_decisions (
                    context TEXT,
                    decision TEXT,
                    user_id INTEGER PRIMARY KEY REFERENCES users(uid) 
                );
                """
            )

            # Check if the prompt library table exists
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = 'context_library'
                );
                """
            )
            prompt_table_exists = cursor.fetchone()[0]

            if not prompt_table_exists:
                cursor.execute(
                    """
                    CREATE TABLE context_library (
                        context TEXT,
                        context_id SERIAL PRIMARY KEY
                    );
                    """
                )

                data: pd.DataFrame = pd.read_json(
                    "context_library.jsonl", lines=True)
                for _, row in data.iterrows():
                    cursor.execute(
                        "INSERT INTO context_library (context, context_id) VALUES (%s, %s)",
                        (row["context"], row["id"])
                    )

    def insert_user(self, user: User):
        with self.connection as cursor:
            try:
                cursor.execute(
                    """
                    INSERT INTO users (name, email, age, done_se, industry_exp, industry_org_name, rate_yourself, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING uid;
                    """,
                    (user.name, user.email, user.age, user.done_se, user.industry_exp,
                     user.industry_org_name, user.rate_yourself, user.created_at)
                )
                uid = cursor.fetchone()[0]
                return uid
            except Exception as e:
                self.connection.__exit__("Exception", e, None)
                raise e

    def insert_experiment_run(self, run: ExperimentRun):
        with self.connection as cursor:
            try:
                cursor.execute(
                    """
                    INSERT INTO experiment_runs (context, user_id, timestamp, 
                    decision1, decision1_rating, decision1_note, decision1_approach, 
                    decision2, decision2_rating, decision2_note, decision2_approach)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING run_id;
                    """,
                    (run.context, run.user_id, run.timestamp,
                     run.decision1, run.decision1_rating, run.decision1_note, run.decision1_approach,
                     run.decision2, run.decision2_rating, run.decision2_note, run.decision2_approach)
                )
                run_id = cursor.fetchone()[0]
                return run_id
            except Exception as e:
                self.connection.__exit__("Exception", e, None)
                raise e

    def get_context_library(self):
        with self.connection as cursor:
            cursor.execute("SELECT * FROM context_library")
            prompts = cursor.fetchall()
            return prompts

    def add_custom_decision(self, context, decision, user_id):
        with self.connection as cursor:
            try:
                cursor.execute(
                    "INSERT INTO custom_decisions (context, decision, user_id) VALUES (%s, %s, %s) RETURNING user_id;",
                    (context, decision, user_id)
                )
                id_ = cursor.fetchone()[0]
                return id_ if id_ else False
            except Exception as e:
                self.connection.__exit__("Exception", e, None)
                raise e


# if __name__ == "__main__":
#     db = Database()
#     db.start()
#     print(db.get_context_library())
#     user = User(
#         uid=0,  # Temporary, since it will be auto-generated in the database
#         name="John Doe",
#         email="anh@njds.com",
#         age=25,
#         done_se="Yes",
#         industry_exp=5,
#         industry_org_name="ABC Inc.",
#         rate_yourself=4,
#         created_at=datetime.now()
#     )
#     uid = db.insert_user(user)
#     print("User ID:", uid)

#     run = ExperimentRun(
#         context="We need to decide on a database for our application.",
#         user_id=uid,
#         timestamp=datetime.now(),
#         decision1="We will use PostgreSQL because it is open-source and has good community support.",
#         decision1_rating=5,
#         decision1_note="No notes",
#         decision1_approach=1,
#         decision2="We will use MySQL because it is widely used and has good performance.",
#         decision2_rating=4,
#         decision2_note="No notes",
#         decision2_approach=2
#     )
#     run_id = db.insert_experiment_run(run)
#     print("Experiment Run inserted successfully.",
#           run_id if run_id else "Failed to insert Experiment Run.")
