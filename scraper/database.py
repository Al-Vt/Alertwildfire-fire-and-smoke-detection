import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
import json

# We load the environment variables
load_dotenv()

# Retrieving the Neon URL from the .env file
DB_CONNECTION_STRING = os.getenv("DATABASE_URL")

class DatabaseManager:
    def __init__(self):
        # Database connection
        try:
            # Each SQL query is validated without having to commit
            self.conn = psycopg2.connect(DB_CONNECTION_STRING)
            self.conn.autocommit = True
        except Exception as e:
            logging.error(f"Error of connexion: {e}")
            raise e

    def insert_image(self, batch_id, camera_name, s3_path):
        """Save a new image"""
        query = """
        INSERT INTO images (batch_id, camera_name, s3_path, status)
        VALUES (%s, %s, %s, 'NEW')
        RETURNING id;
        """
        try:
            with self.conn.cursor() as cur:
                # inject the three values ​​in place of the %s
                cur.execute(query, (batch_id, camera_name, s3_path))
                # take the id
                image_id = cur.fetchone()[0]
                return image_id
        except Exception as e:
            logging.error(f"Error insert_image: {e}")
            return None

    def get_pending_images(self, limit=10):
        """Retrieve the pending images"""
        query = """
        SELECT * FROM images 
        WHERE status = 'NEW' 
        ORDER BY captured_at ASC 
        LIMIT %s;
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Error get_pending_images: {e}")
            return []

    def update_prediction(self, image_id, fire_detected, confidence, bbox):
        """Updates the AI ​​result"""
        query = """
        UPDATE images 
        SET status = 'PROCESSED', 
            fire_detected = %s, 
            confidence = %s, 
            bbox = %s
        WHERE id = %s;
        """
        try:
            # We convert the list to JSON
            bbox_json = json.dumps(bbox)
            with self.conn.cursor() as cur:
                # inject the four values ​​in place of the %s
                cur.execute(query, (fire_detected, confidence, bbox_json, image_id))
        except Exception as e:
            logging.error(f"Error update_prediction: {e}")

    def close(self):
        if self.conn:
            self.conn.close()


    def get_recent_processed_images(self, limit=100):
        """Retrieve the already processed images"""
        query = """
        SELECT * FROM images 
        WHERE status = 'PROCESSED' 
        ORDER BY captured_at ASC 
        LIMIT %s;
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Error get_recent_processed_images: {e}")
            return []
