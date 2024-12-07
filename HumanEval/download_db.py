import psycopg2
import json
from datetime import datetime
from decimal import Decimal

def custom_serializer(obj):
    """
    Custom serializer to handle non-JSON serializable data types.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 string
    elif isinstance(obj, Decimal):
        return float(obj)  # Convert Decimal to float
    raise TypeError(f"Type {type(obj)} not serializable")

def scrape_postgres(db_config):
    """
    Scrapes all tables and their records from a PostgreSQL database.
    
    Args:
        db_config (dict): Configuration dictionary for the database connection.
    Returns:
        dict: A dictionary with table names as keys and rows as values.
    """
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        # Query to get all table names
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        tables = cursor.fetchall()

        # Initialize a dictionary to store all table data
        database_data = {}

        for table in tables:
            table_name = table[0]
            print(f"Fetching data from table: {table_name}")

            # Query all records from the current table
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()

            # Fetch column names for the table
            colnames = [desc[0] for desc in cursor.description]

            # Store data in a list of dictionaries
            database_data[table_name] = [
                dict(zip(colnames, row)) for row in rows
            ]

        # Close cursor and connection
        cursor.close()
        connection.close()

        return database_data

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":

    db_config = {
        'host': 'localhost',
        'dbname': 'llm4adr',
        'user': 'root',
        'password': 'root',
        'port': 5431
    }

    scraped_data = scrape_postgres(db_config)

    # Save the data to a JSON file
    if scraped_data:
        with open("database_dump.json", "w") as f:
            json.dump(scraped_data, f, indent=4, default=custom_serializer)
        print("Database dump saved to 'database_dump.json'.")
