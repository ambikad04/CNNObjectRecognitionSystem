# Real-Time Object Detection System with CNN

This project implements a real-time object detection system using PyTorch and CNNs. The system performs object detection, segmentation, and recognition in real-time using a webcam feed.

## Features

- Object Detection: Locates objects in images using a custom CNN
- Segmentation: Creates precise masks for detected objects using U-Net
- Recognition: Identifies objects using feature extraction and classification
- Vector Matching: Matches detected objects against a product database using FAISS

## Project Structure

```
.
├── models/
│   ├── cnn_detector.py    # Object detection model
│   ├── unet_segmenter.py  # Segmentation model
│   └── recognition_model.py # Recognition model
├── pipeline.py           # Main processing pipeline
├── demo.py              # Real-time demo script
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the real-time demo:
```bash
python demo.py
```

2. Press 'q' to quit the demo.

## Model Training

The models need to be trained before use. You'll need to:

1. Prepare your dataset
2. Train the detector model
3. Train the segmentation model
4. Train the recognition model
5. Build your product database

## Customization

- Modify `num_classes` in `demo.py` to match your dataset
- Adjust confidence thresholds in `pipeline.py`
- Customize the visualization in `demo.py`

## Requirements

- Python 3.7+
- PyTorch 2.0+
- OpenCV
- FAISS
- CUDA (optional, for GPU acceleration)

## License

MIT License 
