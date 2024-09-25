# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routers.image_router import router as image_router

app = FastAPI()

# Configure CORS (allowing access from other domains, including localhost for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict it to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include the image router, which handles the image upload and YOLO processing
app.include_router(image_router, prefix="/image")

# Example root route (for health check or simple response)
@app.get("/")
async def root():
    return {"message": "Welcome to the YOLO Image Processing API"}
