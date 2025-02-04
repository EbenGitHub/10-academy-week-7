import logging
import os
import json
import csv
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient

# Load environment variables
load_dotenv()
api_id = os.getenv("TG_API_ID")
api_hash = os.getenv("TG_API_HASH")
phone = os.getenv("PHONE")

# Set up logging
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "data_scraper.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Load channels from channels.json
def load_channels():
    try:
        with open("../data/raw/channels.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return [channel.split("/")[-1] for channel in data.get("channels", [])]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading channels.json: {e}")
        return []


# Function to get last processed message ID
def get_last_processed_id(channel_username):
    try:
        with open(f"../data/raw/{channel_username}_last_id.json", "r") as f:
            return json.load(f).get("last_id", 0)
    except FileNotFoundError:
        logging.warning(
            f"No last ID file found for {channel_username}. Starting from 0."
        )
        return 0


# Function to save last processed message ID
def save_last_processed_id(channel_username, last_id):
    with open(f"../data/raw/{channel_username}_last_id.json", "w") as f:
        json.dump({"last_id": last_id}, f)
    logging.info(f"Saved last processed ID {last_id} for {channel_username}.")


# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, media_dir, rows):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        last_id = get_last_processed_id(channel_username)
        message_count = 0

        async for message in client.iter_messages(
            entity, reverse=True
        ):  # Ensure older messages come first
            if message_count >= 1000:
                break

            media_path = None
            if message.media:
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)
                logging.info(f"Downloaded media for message ID {message.id}.")

            # Store the row in a list instead of writing immediately
            rows.append(
                [
                    channel_title,
                    channel_username,
                    message.id,
                    message.message,
                    message.date.strftime("%Y-%m-%d %H:%M:%S"),
                    media_path,
                ]
            )
            logging.info(f"Processed message ID {message.id} from {channel_username}.")
            last_id = message.id
            message_count += 1

        save_last_processed_id(channel_username, last_id)
        if message_count == 0:
            logging.info(f"No new messages found for {channel_username}.")

    except Exception as e:
        logging.error(f"Error while scraping {channel_username}: {e}")


# Initialize the client
client = TelegramClient("../logs/scraping_session", api_id, api_hash)


async def main():
    try:
        await client.start(phone)
        logging.info("Client started successfully.")
        media_dir = "../data/raw/photos"
        os.makedirs(media_dir, exist_ok=True)

        csv_path = "../data/raw/scraped_data.csv"
        file_exists = os.path.isfile(csv_path)

        rows = []  # Store rows in a list

        channels = load_channels()
        for channel in channels:
            await scrape_channel(client, channel, media_dir, rows)
            logging.info(f"Scraped data from {channel}.")

        # Write to CSV once all data is collected
        with open(csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write headers only if the file is new
            if not file_exists:
                writer.writerow(
                    [
                        "Channel Title",
                        "Channel Username",
                        "ID",
                        "Message",
                        "Date",
                        "Media Path",
                    ]
                )

            writer.writerows(rows)  # Write all collected rows at once

    except Exception as e:
        logging.error(f"Error in main function: {e}")


if __name__ == "__main__":
    asyncio.run(main())
