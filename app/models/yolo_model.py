# app/models/yolo_model.py
from ultralytics import YOLO

class YoloModel:
    def __init__(self):
        # Load a pre-trained YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # YOLOv8n model (smallest version)

    def predict(self, image_path: str):
        try:
            # Use YOLO to make predictions on the image
            results = self.model(image_path)
            return results.pandas().xyxy[0].to_dict(orient="records")
        except Exception as e:
            raise ValueError(f"Error processing image with YOLO: {str(e)}")
