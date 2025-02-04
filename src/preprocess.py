import os
import re
import json
import string
import logging
import pandas as pd

from datetime import datetime


# Ensure logs folder exists
os.makedirs("../logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/data_cleaning.log"),
        logging.StreamHandler(),
    ],
)


def extract_messages(data_paths):
    """
    Extracts messages from JSON files and returns a Pandas DataFrame.

    Parameters:
        data_paths (list): List of file paths to JSON message data.

    Returns:
        pd.DataFrame: DataFrame containing extracted message data.
    """
    try:
        messages_data = []

        for data_path in data_paths:
            channel_name = data_path.split("/")[-2]  # Extract channel name
            channel_dir = f"../data/raw/scraped_data/{channel_name}/"

            with open(data_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logging.info(f"✅ Loaded JSON file: {data_path}")

            for message in data.get("messages", []):
                # Combine text from different parts
                combined_text = "".join(
                    part["text"] if isinstance(part, dict) and "text" in part else part
                    for part in message.get("text", [])
                ).strip()

                # Extract media path, width, and height
                photo = message.get("photo", None)
                media_path = f"{channel_dir}{photo}" if photo else None
                width = message.get("width", None)
                height = message.get("height", None)

                # Append processed message data
                messages_data.append(
                    {
                        "channel_name": data.get("name", ""),
                        "channel_address": channel_name,
                        "channel_id": data.get("id", ""),
                        "message_id": message.get("id", ""),
                        "date": datetime.strptime(message["date"], "%Y-%m-%dT%H:%M:%S"),
                        "message": combined_text,
                        "media_path": media_path,
                        "width": width,
                        "height": height,
                    }
                )

            logging.info(
                f"✅ Extracted {len(data.get('messages', []))} messages from {channel_name}"
            )

        df = pd.DataFrame(messages_data)
        logging.info(
            f"✅ Successfully created DataFrame with {df.shape[0]} rows and {df.shape[1]} columns"
        )

        return df

    except Exception as e:
        logging.error(f"❌ Error in extract_messages: {e}")
        raise


def clean_dataframe(df):
    """
    Removes emojis and keeps only alphanumeric characters, punctuation, and spaces from the 'Message' column.
    Adds a 'cleaned_message' column with the cleaned text.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'Message' column to be cleaned.

    Returns:
        pd.DataFrame: DataFrame with an additional 'cleaned_message' column containing the cleaned text.
    """
    try:
        # Check if the 'Message' column exists in the DataFrame
        if "message" not in df.columns:
            raise ValueError("❌ 'Message' column not found in DataFrame.")

        # Clean the 'Message' column and create the 'cleaned_message' column
        df["cleaned_message"] = df["message"].apply(
            lambda x: "".join(
                re.findall(
                    rf"[\u1200-\u137F{string.ascii_letters}{string.digits}{string.punctuation}\s]+",
                    x,
                )
            )
        )

        logging.info(
            f"✅ Successfully cleaned 'message' column and created 'cleaned_message' column."
        )
        return df

    except Exception as e:
        logging.error(f"❌ Error in clean_text: {e}")
        raise


def save_cleaned_data(df, output_path):
    """Save cleaned data to a CSV file."""
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"✅ Cleaned data saved to '{output_path}'.")
        print(f"✅ Cleaned data saved to '{output_path}'.")
    except Exception as e:
        logging.error(f"❌ Error saving cleaned data: {e}")
        raise
