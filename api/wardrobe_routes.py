from fastapi import APIRouter
from services.wardrobe_service import get_wardrobe_recs, flatten_recommendations, get_embedding_from_huggingface
from services.s3_service import S3Service
from fastapi.responses import JSONResponse
from models.request_models import TextRequest, MatchWardrobeRequest
import numpy as np


router = APIRouter()
s3_service = S3Service() 


@router.post("/vectorize")
async def vectorize_text(request: TextRequest):
    """Vectorize the text."""
    vector = await get_embedding_from_huggingface(request.text)  
    if vector is not None and isinstance(vector, np.ndarray):
        vector = vector.tolist()
    return JSONResponse(content={"vector": vector})

@router.post("/match")
async def match_wardrobe(request: MatchWardrobeRequest):
    """Match the user's wardrobe to the recommendations."""
    category_results = {}
    for category, items in request.recommendations.items():
        if not items:
            continue
        
        # Fix: items appears to be a dictionary with a single key 'Suggestions'
        for item_category, item_list in items.items():
            results = []
            
            # Now item_list is a list of dictionaries, each with 'Clothing Type' and 'Color'
            for recs in item_list:
                # Only process if it's a dictionary containing clothing items
                if isinstance(recs, dict) and "Clothing Type" in recs:
                    clothing_type = recs.get("Clothing Type", "")
                    color = recs.get("Color", "")                  
                    input_text = f"{color} {clothing_type}"
                    res = await get_wardrobe_recs(input_text, request.user_id)
                    # if(res.get("category") in category):
                    results.append(res)
            
            if results:
                if category not in category_results:
                    category_results[category] = {}
                category_results[category][item_category] = results

    print(category_results)

    return JSONResponse(content={
        "recommendations": [flatten_recommendations(category_results)]
    })   
