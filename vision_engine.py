import os
import cv2
from ultralytics import YOLO

class VisionEngine:
    def __init__(self, model_weight="yoloe-26n-seg.pt"):
        """
        Initializes the YOLOE-26 zero-shot open-vocabulary model.
        Leverages native end-to-end, NMS-free inference.
        """
        print(f"Loading YOLOE-26 engine with {model_weight}...")
        # Load the pre-trained open-vocabulary model
        self.model = YOLO(model_weight)
        
    