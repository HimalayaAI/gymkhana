import psycopg2
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def init_database():
    """
    Initialize the PostgreSQL database:
    1. Connect to 'postgres' database to create the target database if it doesn't exist.
    2. Connect to the target database and apply the schema.
    """
    db_name = os.getenv("DB_NAME", "gymkhana")
    db_user = os.getenv("DB_USER", "db_user")
    db_password = os.getenv("DB_PASSWORD", "db_pwd@123")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5433")

    print(f"Initializing database '{db_name}' on {db_host}:{db_port}...")

    try:
        # Step 1: Create database if missing
        conn = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            dbname="postgres",
            port=db_port
        )
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"CREATE DATABASE {db_name}")
                print(f"Database '{db_name}' created.")
            except psycopg2.errors.DuplicateDatabase:
                print(f"Database '{db_name}' already exists.")
        conn.close()
    except Exception as e:
        print(f"Error checking/creating database: {e}")
        # Continue anyway, let connection to target DB fail if it truly doesn't exist

    try:
        # Step 2: Apply schema
        conn = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            dbname=db_name,
            port=db_port
        )

        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "gymkhana/core/services/storage/schema.sql"
        )

        if not os.path.exists(schema_path):
            print(f"Error: Schema file not found at {schema_path}")
            sys.exit(1)

        print(f"Applying schema from {schema_path}...")
        with open(schema_path, "r") as f:
            schema = f.read()

        with conn.cursor() as cursor:
            cursor.execute(schema)

        conn.commit()
        conn.close()
        print("Schema applied successfully!")
    except Exception as e:
        print(f"Error applying schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
