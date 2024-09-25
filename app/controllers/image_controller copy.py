# app/controllers/image_controller.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil
from app.models.yolo_model import YoloModel

router = APIRouter()

upload_folder = "app/uploads/"
yolo_model = YoloModel()

# Ensure the upload directory exists
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@router.post("/upload/")
async def upload_and_process_image(file: UploadFile = File(...)):
    # Check valid image file extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")
    
    file_path = os.path.join(upload_folder, file.filename)
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

    # Run YOLO inference on the image
    try:
        yolo_results = yolo_model.predict(file_path)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    
    return {"yolo_results": yolo_results, "file_path": f"/uploads/{file.filename}"}
