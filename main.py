from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import image_captioning, wardrobe_routes
from dotenv import load_dotenv
from utils.database import check_connection
from contextlib import asynccontextmanager


load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify connections
    mongo_connected = check_connection()
    if mongo_connected:
        print("✅ Successfully connected to MongoDB")
    else:
        print("❌ Failed to connect to MongoDB")

    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(image_captioning.router, prefix="/api/caption", tags=["Image Captioning"])
app.include_router(wardrobe_routes.router, prefix="/api/wardrobe", tags=["Wardrobe"]) 

@app.get("/health")
async def health():
    return {"message": "Fashion Recommendation ML API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)




