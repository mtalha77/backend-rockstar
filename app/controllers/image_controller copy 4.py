# app/controllers/image_controller.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil
from roboflow import Roboflow  # Import the Roboflow SDK
from dotenv import load_dotenv
import logging

# Load environment variables from the .env file
load_dotenv()

# Retrieve Roboflow API details from environment variables
ROBOFLOW_API_KEY = os.getenv("API_KEY")  # Your API key from the .env file
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")  # Your model endpoint
VERSION = int(os.getenv("MODEL_VERSION", 2))  # Your model version (as an integer)

# Initialize Roboflow using the API key
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(MODEL_ENDPOINT)
model = project.version(VERSION).model  # Load the specific version of the model

# Initialize FastAPI router
router = APIRouter()

# Define paths
upload_folder = "uploads/"

# Ensure the upload directory exists
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Enhanced error logging setup
logging.basicConfig(level=logging.INFO)

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
        logging.info(f"Image saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to upload image: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

    # Perform inference using Roboflow SDK
    try:
        logging.info(f"Performing inference on {file_path}...")  # Debugging line
        
        # Roboflow SDK will return results in JSON format
        roboflow_response = model.predict(file_path, confidence=40, overlap=30).json()  # No need for `.pandas()`
        logging.info(f"Inference results: {roboflow_response}")  # Debugging line

        # Ensure roboflow_response contains the expected data
        if 'predictions' not in roboflow_response:
            logging.error("Predictions missing in response")
            raise HTTPException(status_code=500, detail="Error: Predictions missing in Roboflow response")
        
    except Exception as e:
        logging.error(f"Error during Roboflow inference: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error during Roboflow inference: {str(e)}")

    # Return the results and file path for the uploaded image
    try:
        response = {"roboflow_results": roboflow_response, "file_path": f"/uploads/{file.filename}"}
        logging.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error returning response: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error returning response: {str(e)}")
