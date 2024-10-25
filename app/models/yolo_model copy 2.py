import cv2
from ultralytics import YOLO
class YoloModel:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
    def predict(self, image_path: str):
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Image at path {image_path} could not be loaded")
            original_height, original_width = image_bgr.shape[:2]
            results = self.model(image_bgr)[0]
            return_dict = {"masks": [], "img_height": original_height, "img_width": original_width}
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x = int(x1)
                    y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    mask_info = {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "filled": False
                    }
                    return_dict["masks"].append(mask_info)
            else:
                return_dict["masks"] = None
            return return_dict
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
