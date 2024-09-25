import cv2
import torch
from ultralytics import YOLO

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
            return_dict = {"box": {}, "masks": {}}

            # Extract boxes information (bounding box coordinates, confidence, and class)
            if results.boxes is not None:
                return_dict["box"]["xyxy"] = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
                return_dict["box"]["confidence"] = results.boxes.conf.cpu().numpy()  # Confidence scores
                return_dict["box"]["class_id"] = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            else:
                return_dict["box"] = None

            # Check if segmentation masks are present
            if results.masks is not None:
                return_dict["masks"]["values"] = results.masks.data.cpu().numpy()  # Segmentation masks
                return_dict["masks"]["count"] = len(results.masks.data)  # Number of masks
                return_dict["masks"]["img_height"] = original_height  # Original image height
                return_dict["masks"]["img_width"] = original_width  # Original image width
            else:
                return_dict["masks"] = None

            # Return the results dictionary
            return return_dict

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
