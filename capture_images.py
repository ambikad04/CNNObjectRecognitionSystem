import cv2
import os
import json
import numpy as np
from datetime import datetime
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def create_annotation_file(images_dir, output_file):
    """Create an empty annotation file with the required structure."""
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "phone"
            }
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)

def load_model():
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model

def detect_objects(model, frame, device):
    # Convert frame to tensor
    image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image)
    
    # Get boxes, scores, and labels
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Filter predictions with confidence > 0.5
    mask = scores > 0.5
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels

def get_next_image_number(directory):
    """Get the next available image number in the directory."""
    existing_files = [f for f in os.listdir(directory) if f.startswith('phone') and f.endswith('.jpg')]
    if not existing_files:
        return 1
    
    numbers = []
    for file in existing_files:
        try:
            num = int(file.replace('phone', '').replace('.jpg', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def capture_and_annotate():
    # Create directories if they don't exist
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    
    # Create annotation files if they don't exist
    if not os.path.exists('data/train/annotations.json'):
        create_annotation_file('data/train', 'data/train/annotations.json')
    if not os.path.exists('data/val/annotations.json'):
        create_annotation_file('data/val', 'data/val/annotations.json')
    
    # Load existing annotations
    with open('data/train/annotations.json', 'r') as f:
        train_annotations = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model()
    model.to(device)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set window name
    window_name = "Capture and Annotate"
    cv2.namedWindow(window_name)
    
    # Initialize variables
    current_image = None
    current_image_path = None
    status_text = "Press 'c' to capture an image"
    selected_box = None
    
    print("\nInstructions:")
    print("1. Press 'c' to capture an image")
    print("2. Click on a detected object to select it")
    print("3. Press 's' to save the annotation")
    print("4. Press 'q' to quit")
    print("\nStatus messages will appear in the window")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Detect objects in the frame
        boxes, scores, labels = detect_objects(model, frame, device)
        
        # Draw all detected boxes
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if box is selected_box else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add status text
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('c'):
            # Get next image number
            next_num = get_next_image_number('data/train')
            current_image_path = f'data/train/phone{next_num}.jpg'
            cv2.imwrite(current_image_path, frame)
            current_image = frame.copy()
            status_text = "Image captured! Click on an object to select it"
            print(f"\nCaptured image: {current_image_path}")
            
        elif key == ord('s') and current_image is not None and selected_box is not None:
            # Save annotation
            image_id = len(train_annotations['images']) + 1
            annotation_id = len(train_annotations['annotations']) + 1
            
            # Add image info
            train_annotations['images'].append({
                'id': image_id,
                'file_name': os.path.basename(current_image_path),
                'width': frame.shape[1],
                'height': frame.shape[0]
            })
            
            # Add annotation
            x1, y1, x2, y2 = map(int, selected_box)
            w = x2 - x1
            h = y2 - y1
            
            train_annotations['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [x1, y1, w, h]
            })
            
            # Save updated annotations
            with open('data/train/annotations.json', 'w') as f:
                json.dump(train_annotations, f, indent=4)
            
            print(f"Saved annotation for image {image_id}")
            status_text = "Annotation saved! Press 'c' to capture next image"
            
            # Reset for next capture
            current_image = None
            selected_box = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_box, status_text
        
        if event == cv2.EVENT_LBUTTONDOWN and current_image is not None:
            # Check if click is inside any box
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_box = box
                    status_text = "Object selected! Press 's' to save"
                    break
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_annotate() 