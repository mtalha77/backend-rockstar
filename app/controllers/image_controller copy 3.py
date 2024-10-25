# app/controllers/image_controller.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve Roboflow API details from environment variables
ROBOFLOW_API_KEY = os.getenv("API_KEY")  # Your API key from the .env file
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")  # Your model endpoint
VERSION = int(os.getenv("MODEL_VERSION", 2))  # Your model version (as an integer)

# Initialize Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(MODEL_ENDPOINT)
model = project.version(VERSION).model

# Initialize FastAPI router
router = APIRouter()

# Define paths
upload_folder = "uploads/"

# Ensure the upload directory exists
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Route to handle image upload and send to Roboflow for inference
@router.post("/upload/")
async def upload_and_process_image(file: UploadFile = File(...)):
    # Check for valid image file extensions
    valid_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")
    
    # Save the uploaded image locally
    file_path = os.path.join(upload_folder, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

    # Perform inference using Roboflow SDK
    try:
        # Confidence and overlap settings can be customized
        roboflow_response = model.predict(file_path, confidence=40, overlap=30).json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Roboflow inference: {str(e)}")

    # Return the results and file path for the uploaded image
    return {"roboflow_results": roboflow_response, "file_path": f"/uploads/{file.filename}"}
