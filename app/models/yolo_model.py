import cv2
import torch
from ultralytics import YOLO
import json
import numpy as np

class YoloModel:
    def __init__(self):
        self.model = YOLO('best.pt')
    def predict(self, image_path: str):
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Image at path {image_path} could not be loaded")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.model(image_rgb)[0]
            return_dict = {"masks": []}
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                for mask in masks:
                    mask_points = np.argwhere(mask > 0)
                    swapped_mask_points = [[y, x] for x, y in mask_points]
                    mask_points_list = swapped_mask_points
                    return_dict["masks"].append(mask_points_list)
            return return_dict
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

