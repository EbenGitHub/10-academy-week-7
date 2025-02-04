import logging
import requests
import os
import sqlite3
import torch
import cv2
from pathlib import Path
from yolov5 import YOLOv5

# Setting up logging
logging.basicConfig(level=logging.INFO)


def setup_yolo_model(model_path="yolov5s.pt"):
    """Load the pre-trained YOLO model."""
    try:
        # Load model (PyTorch version)
        model = YOLOv5(model_path)
        logging.info(f"✅ YOLO model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        raise


def download_images_from_telegram(channel_url, download_dir="images"):
    """Download images from the provided Telegram channel."""
    try:
        # Replace this with actual scraping from the channel
        os.makedirs(download_dir, exist_ok=True)

        # Example: Downloading a few images (this part should be tailored for scraping Telegram)
        for i in range(5):
            image_url = f"{channel_url}/image_{i}.jpg"  # Replace with actual image URL
            image_data = requests.get(image_url).content
            image_path = os.path.join(download_dir, f"image_{i}.jpg")

            with open(image_path, "wb") as f:
                f.write(image_data)

            logging.info(f"✅ Image {i} downloaded.")
    except Exception as e:
        logging.error(f"❌ Error downloading images: {e}")
        raise


def detect_objects_in_images(model, image_dir="images"):
    """Run object detection on the images in the provided directory using YOLO."""
    detections = []
    try:
        for image_path in Path(image_dir).glob("*.jpg"):
            # Load image
            img = cv2.imread(str(image_path))

            # Perform detection
            results = model(img)
            detections.append(
                {
                    "image": image_path.name,
                    "boxes": results.xywh[0].tolist(),
                    "labels": results.names[results.xywh[0][:, 5].astype(int)].tolist(),
                    "confidences": results.xywh[0][:, 4].tolist(),
                }
            )

            logging.info(f"✅ Detected objects in {image_path.name}")
    except Exception as e:
        logging.error(f"❌ Error detecting objects: {e}")
        raise
    return detections


def store_detection_results(connection, detections):
    """Store object detection results in SQLite database."""
    try:
        insert_query = """
        INSERT INTO object_detections (image_name, label, confidence, x_center, y_center, width, height) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        with connection:
            for detection in detections:
                for box, label, confidence in zip(
                    detection["boxes"], detection["labels"], detection["confidences"]
                ):
                    x_center, y_center, width, height = box[:4]
                    connection.execute(
                        insert_query,
                        (
                            detection["image"],
                            label,
                            confidence,
                            x_center,
                            y_center,
                            width,
                            height,
                        ),
                    )

        logging.info(f"✅ {len(detections)} detection results stored in database.")
    except Exception as e:
        logging.error(f"❌ Error storing detection results: {e}")
        raise


def create_database_table(connection):
    """Create the object_detections table if it does not exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS object_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        label TEXT,
        confidence REAL,
        x_center REAL,
        y_center REAL,
        width REAL,
        height REAL
    );
    """
    try:
        with connection:
            connection.execute(create_table_query)
        logging.info("✅ Table 'object_detections' created successfully.")
    except Exception as e:
        logging.error(f"❌ Error creating table: {e}")
        raise


def main():
    # Set up database connection
    connection = sqlite3.connect("detections.db")

    # Set up YOLO model
    model = setup_yolo_model()

    # Download images from Telegram channel
    channel_url = "https://t.me/lobelia4cosmetics"  # Replace with actual channel URL
    download_images_from_telegram(channel_url)

    # Detect objects in the downloaded images
    detections = detect_objects_in_images(model)

    # Store the detection results in the database
    create_database_table(connection)
    store_detection_results(connection, detections)

    # Close the database connection
    connection.close()
    logging.info("✅ Process completed successfully.")


if __name__ == "__main__":
    main()
