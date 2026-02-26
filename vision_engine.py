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
        
    def process_image(self, image_path, text_prompts, output_path):
        """
        Runs zero-shot prompt-driven detection and segmentation on an image.
        
        Args:
            image_path (str): Path to the uploaded input image.
            text_prompts (str or list): Comma-separated string or list of objects to detect.
            output_path (str): Path to save the annotated output image.
            
        Returns:
            bool: True if processing was successful, False otherwise.
        """
        try:
            # Parse comma-separated prompts into a list format
            if isinstance(text_prompts, str):
                prompts_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
            else:
                prompts_list = text_prompts
                
            #             # Dynamically set the vocabulary for zero-shot detection.
            # YOLOE-26 handles the text embeddings automatically here.
            self.model.set_classes(prompts_list)
            
            # Run inference. The NMS-free architecture ensures this is fast and deterministic.
            # We set a baseline confidence threshold, but this can be adjusted.
            results = self.model.predict(source=image_path, save=False, conf=0.15)
            
            # Extract the annotated image array directly from the Ultralytics results object
            annotated_frame = results[0].plot()
            
            # Save the processed output using OpenCV
            cv2.imwrite(output_path, annotated_frame)
            
            print(f"Successfully processed and saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing image with YOLOE-26: {e}")
            return False