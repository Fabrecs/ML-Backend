from fastapi import APIRouter
from services.image_captioning_service import generate_caption
from models.request_models import ImageURLRequest

router = APIRouter()

@router.post("/")
async def caption_image(request: ImageURLRequest):
    """Generate an image caption from a clothing image."""
    return await generate_caption(request.image_url)
