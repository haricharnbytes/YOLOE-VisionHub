# YOLOE-VisionHub

**YOLOE-VisionHub** is a simple Flask-based web application that leverages the **YOLOE-26** model for zero-shot object detection and segmentation. The application allows users to upload images, specify object detection prompts, and view annotated results.

## Features

- **Zero-shot Object Detection**: Detects objects in images based on user-specified prompts (e.g., "laptop", "person").
- **Flask Web Interface**: Upload images and view results through a simple web interface.
- **Real-time Processing**: Uses YOLOE-26 for fast, real-time inference and object detection.

## Requirements

- **Python 3.8+**
- **OpenCV**: For image handling and saving results.
- **Ultralytics YOLO**: YOLOE model for object detection.

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt

