from pymongo import MongoClient
from dotenv import load_dotenv
import os
import uuid
import numpy as np
from utils.s3_utils import generate_signed_urls
from services.text_vectorization_service import get_text_vector
# Load environment variables
load_dotenv()

# Connect to MongoDB
MONGO_URL = os.getenv("MONGO_URL")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
client = MongoClient(MONGO_URL)
db = client["fabrecsai"]
collection = db["wardrobe"]

def flatten_recommendations(category_results):
    """
    Convert categorized recommendations into a flat array of objects.
    
    Args:
        category_results (dict): Dictionary containing categorized recommendations
        
    Returns:
        list: Flat array of recommendation objects
    """
    flat_recommendations = []
    recommendations = category_results.get("recommendations", {})
    
    for category, items_list in recommendations.items():
        # Handle the case where items is a list of lists
        if isinstance(items_list, list):
            for items in items_list:
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            # Add category to each item
                            item_with_category = item.copy()
                            item_with_category["category"] = category
                            flat_recommendations.append(item_with_category)
    
    return flat_recommendations

def add_wardrobe_item(user_id: str, image_url: str, caption: str, category: str):
    item_id = str(uuid.uuid4())
    caption_embedding = get_text_vector(caption)
    if caption_embedding is not None and isinstance(caption_embedding, np.ndarray):
        caption_embedding = caption_embedding.tolist()
    item = {
        "_id": item_id,
        "user_id": user_id,
        "image_url": image_url,  
        "caption": caption,
        "caption_embedding": caption_embedding,
        "category": category
    }
    collection.insert_one(item)
    return item

def get_user_wardrobe(user_id: str):
    wardrobe_items = list(collection.find({"user_id": user_id}))
    
    # Convert ObjectId to string
    for item in wardrobe_items:
        item["_id"] = str(item["_id"])
    
    # Collect all image URLs
    image_urls = [item["image_url"] for item in wardrobe_items]
    
    # Generate signed URLs for all images at once
    if image_urls:
        signed_urls = generate_signed_urls(urls=image_urls, client_method='get_object')
        
        # Update items with signed URLs
        for i, item in enumerate(wardrobe_items):
            item["image_url"] = signed_urls[i]
    
    return wardrobe_items

def delete_wardrobe_item(item_id: str, user_id: str):
    result = collection.delete_one({"_id": item_id, "user_id": user_id})
    return result.deleted_count == 1

async def get_wardrobe_recs(input_caption, user_id: str):
    try:
        embedding = await get_embedding_from_huggingface(input_caption)
        documents = await findSimilarDocuments(embedding, user_id)
        return documents
    except Exception as err:
        print(err)
    
async def get_embedding_from_huggingface(input_caption):
    try:
        return get_text_vector(input_caption)
    except Exception as err:
        print(err)

def numpy_to_list(obj):
    """Convert numpy arrays to lists for MongoDB compatibility"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

async def findSimilarDocuments(embedding, user_id: str):
    embedding = numpy_to_list(embedding)
    pipeline = [
        {
            "$vectorSearch": {
                "numCandidates": 100,
                "queryVector": embedding,
                "path": "caption_embedding",
                "limit": 2,
                "index": "vector_index",
                "filter": {
                    "user_id": user_id  
                }
            }
        }
    ]
    cursor = collection.aggregate(pipeline)
    documents = list(cursor)
    
    # Convert ObjectId to string for JSON serialization
    for doc in documents:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
    
    # Generate signed URLs for all image URLs
    if documents:
        image_urls = [doc.get("image_url") for doc in documents if "image_url" in doc]
        if image_urls:
            signed_urls = generate_signed_urls(urls=image_urls, client_method='get_object')
            
            # Update documents with signed URLs
            url_index = 0
            for doc in documents:
                if "image_url" in doc:
                    doc["image_url"] = signed_urls[url_index]
                    url_index += 1
                    doc["caption_embedding"] = None
    
    return documents



    