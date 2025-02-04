import os
import uuid
import logging
import sqlite3
from dotenv import load_dotenv
import pandas as pd

# Ensure logs folder exists
os.makedirs("../logs", exist_ok=True)

# Configure logging to write to file & display in Jupyter Notebook
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/database_setup.log"),  # Log to file
        logging.StreamHandler(),  # Log to Jupyter Notebook
    ],
)

# Load environment variables
load_dotenv("../.env")

# Database file path
DB_PATH = "../data/processed/telegram_messages.db"


def get_db_connection():
    """Create and return database connection."""
    try:
        connection = sqlite3.connect(DB_PATH)
        logging.info("✅ Successfully connected to the SQLite database.")
        return connection
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        raise


def create_table(connection):
    """Create telegram_messages table if it does not exist with db_id as primary key."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS telegram_messages (
        db_id TEXT PRIMARY KEY,  -- Use TEXT for UUID as primary key
        channel_name TEXT,
        channel_address TEXT,
        channel_id INTEGER,
        message_id INTEGER,
        date TIMESTAMP,
        message TEXT,
        cleaned_message TEXT,
        media_path TEXT,
        width REAL,
        height REAL
    );
    """
    try:
        with connection:
            connection.execute(create_table_query)
        logging.info("✅ Table 'telegram_messages' created successfully.")
    except Exception as e:
        logging.error(f"❌ Error creating table: {e}")
        raise


def insert_data(connection, cleaned_df):
    """Inserts cleaned Telegram data into SQLite database with UUID as db_id."""
    try:
        # Convert NaT timestamps to None (NULL in SQL)
        cleaned_df["date"] = cleaned_df["date"].apply(
            lambda x: None if pd.isna(x) else str(x)
        )

        insert_query = """
        INSERT INTO telegram_messages 
        (db_id, channel_name, channel_address, channel_id, message_id, date, message, cleaned_message, media_path, width, height) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(db_id) DO NOTHING;
        """

        with connection:
            for _, row in cleaned_df.iterrows():
                # Generate a unique UUID for each message
                unique_db_id = str(uuid.uuid4())

                # Ensure width and height are floats, set to None if NaN
                width = float(row["width"]) if not pd.isna(row["width"]) else None
                height = float(row["height"]) if not pd.isna(row["height"]) else None

                # Debug log to ensure data is being inserted
                logging.info(f"Inserting: {unique_db_id} - {row['date']}")

                connection.execute(
                    insert_query,
                    (
                        unique_db_id,  # Use the generated UUID as db_id
                        row["channel_name"],
                        row["channel_address"],
                        row["channel_id"],
                        row["message_id"],
                        row["date"],
                        row["message"],
                        row["cleaned_message"],
                        row["media_path"],
                        width,
                        height,
                    ),
                )

        logging.info(f"✅ {len(cleaned_df)} records inserted into SQLite database.")
    except Exception as e:
        logging.error(f"❌ Error inserting data: {e}")
        raise
