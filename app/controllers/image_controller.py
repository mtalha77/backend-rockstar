# app/controllers/image_controller.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
import os
import shutil
from ultralytics import YOLO  # Import the YOLOv8 model from ultralytics
import logging

# Initialize FastAPI router
router = APIRouter()

# Define paths
upload_folder = "uploads/"
model_path = "models/best.pt"  # Path to the best.pt model file

# Ensure the upload directory exists
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Enhanced error logging setup
logging.basicConfig(level=logging.INFO)

# Load the YOLOv8 model (best.pt)
try:
    model = YOLO(model_path)  # Load the best.pt model
    logging.info(f"Loaded model from {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading YOLO model: {str(e)}")


# Route to handle image upload and send to YOLOv8 for inference
@router.post("/upload/")
async def upload_and_process_image(
    file: UploadFile = File(...),
    sample_prediction: str = Form(...)
    ):

    if sample_prediction and sample_prediction.lower() == "true":
        return {"Return SMS": "return it from Image controller"}

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

    # Perform inference using YOLOv8 and best.pt
    try:
        logging.info(f"Performing inference on {file_path}...")  # Debugging line

        # Perform inference on the uploaded image
        results = model.predict(file_path)

        # Extract predictions
        predictions = results[0].boxes.xyxy  # Bounding boxes
        confidences = results[0].boxes.conf  # Confidence scores
        classes = results[0].boxes.cls  # Class indices
        
        # Convert to readable format
        prediction_data = []
        for i in range(len(predictions)):
            prediction_data.append({
                "class": int(classes[i].item()),
                "confidence": confidences[i].item(),
                "bbox": predictions[i].tolist()  
            })

        logging.info(f"Inference results: {prediction_data}")  # Debugging line
        
    except Exception as e:
        logging.error(f"Error during YOLOv8 inference: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error during YOLOv8 inference: {str(e)}")

    # Return the results and file path for the uploaded image
    try:
        response = {"predictions": prediction_data, "file_path": f"/uploads/{file.filename}"}
        logging.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error returning response: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error returning response: {str(e)}")
