from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB connection string
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")

# Global MongoDB client
mongo_client = MongoClient(MONGO_URL)

# Database reference
db = mongo_client["test"]  # Using the "test" database as in wardrobe_service

# This function can be called to check the MongoDB connection
def check_connection():
    try:
        # The ismaster command is cheap and does not require auth
        mongo_client.admin.command('ismaster')
        return True
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False 