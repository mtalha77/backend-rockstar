import cv2
import torch
from ultralytics import YOLO
import json
import numpy as np

class YoloModel:
    def __init__(self):
        # Load a pre-trained YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # You can replace this with any YOLOv8 model

    def predict(self, image_path: str):
        try:
            # Read the image using OpenCV (BGR format)
            image_bgr = cv2.imread(image_path)

            if image_bgr is None:
                raise ValueError(f"Image at path {image_path} could not be loaded")

            # Convert image from BGR to RGB format
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Get original image dimensions
            original_height, original_width = image_bgr.shape[:2]

            # Run the model on the RGB image
            results = self.model(image_rgb)[0]  # Get the first result from the list

            # Prepare the return dictionary
            return_dict = {"boxes": [], "masks": []}

            # Process bounding boxes (if available)
            if results.boxes is not None:
                # Extract bounding box coordinates, confidence, and class IDs
                xyxy_boxes = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
                for box in xyxy_boxes:
                    x1, y1, x2, y2 = box
                    # Convert to format [x, y, width, height]
                    width = x2 - x1
                    height = y2 - y1
                    return_dict["boxes"].append([int(x1), int(y1), int(width), int(height)])

            # Process segmentation masks (if available)
            if results.masks is not None:
                # Convert mask data into a JSON-friendly format
                masks = results.masks.data.cpu().numpy()
                for mask in masks:
                    # Flatten mask and send as a list of points (coordinates where the mask is present)
                    mask_points = np.argwhere(mask > 0)  # Get all mask points
                    mask_points_list = mask_points.tolist()  # Convert to a regular list
                    return_dict["masks"].append(mask_points_list)

            # Return the results dictionary as JSON
            return json.dumps(return_dict)

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
