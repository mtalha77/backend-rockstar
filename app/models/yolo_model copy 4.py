import cv2
import torch
from ultralytics import YOLO
import json
import numpy as np

class YoloModel:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def predict(self, image_path: str):
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Image at path {image_path} could not be loaded")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            original_height, original_width = image_bgr.shape[:2]
            results = self.model(image_rgb)[0]
            return_dict = {"masks": []}
            if results.boxes is not None:
                xyxy_boxes = results.boxes.xyxy.cpu().numpy()
                for box in xyxy_boxes:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    return_dict["boxes"].append([int(x1), int(y1), int(width), int(height)])
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                for mask in masks:
                    mask_points = np.argwhere(mask > 0)  
                    mask_points_list = mask_points.tolist() 
                    return_dict["masks"].append(mask_points_list)
            return return_dict

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
