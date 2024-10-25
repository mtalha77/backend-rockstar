# app/controllers/image_controller.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil
import requests
from dotenv import load_dotenv
import base64

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key and model URL from environment variables
ROBOFLOW_API_KEY = os.getenv("API_KEY")
ROBOFLOW_MODEL_URL = os.getenv("MODEL_URL")

# Initialize router
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
    
    # Convert the image to base64
    try:
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the base64 image data for sending in the request
        image_data = {
            "image": encoded_image
        }

        # Send the base64 image to Roboflow for inference
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(f"{ROBOFLOW_MODEL_URL}?api_key={ROBOFLOW_API_KEY}", json=image_data, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to process image with Roboflow")
        
        roboflow_results = response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Roboflow inference: {str(e)}")
    
    # Return the inference results and file path
    return {"roboflow_results": roboflow_results, "file_path": f"/uploads/{file.filename}"}
