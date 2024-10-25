# app/routers/image_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil
from app.models.yolo_model import YoloModel

# Initialize router
router = APIRouter()

# Define paths
upload_folder = "uploads/"
yolo_model = YoloModel()  # Initialize YOLO model

# Ensure the upload directory exists
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Route to handle image upload and YOLO processing
@router.post("/upload/")
async def upload_and_process_image(file: UploadFile = File(...)):
    # Check for valid image file extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")
    
    # Save the uploaded image
    file_path = os.path.join(upload_folder, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

    # Run YOLO model for detection
    try:
        yolo_results = yolo_model.predict(file_path)
        print('yolo_results')
        print(yolo_results)
        return {"yolo_results": str(yolo_results)}
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))

    # Return YOLO results
    # return {"yolo_results": yolo_results, "file_path": f"/uploads/{file.filename}"}
