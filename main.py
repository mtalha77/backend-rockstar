from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers.image_router import router as image_router

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limit to 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # Process the file (e.g., save to disk, process with YOLO, etc.)
    return {"filename": file.filename}

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include the image router
app.include_router(image_router, prefix="/image")

@app.get("/")
async def root():
    return {"message": "Welcome to the YOLO Image Processing API"}
