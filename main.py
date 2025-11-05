import cv2
import numpy as np
from pipeline import ObjectDetectionPipeline
import torch
import json
import time
from datetime import datetime

class ObjectDetectionApp:
    def __init__(self):
        # Initialize pipeline
        self.num_classes = 80  # COCO dataset number of classes
        self.pipeline = ObjectDetectionPipeline(self.num_classes)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Set window properties
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', 1280, 720)
        
        # Initialize FPS calculation variables
        self.prev_time = 0
        self.fps = 0
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.show_mask = True
        self.show_fps = True
        
    def calculate_fps(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
    def draw_results(self, frame, results):
        # Draw FPS if enabled
        if self.show_fps:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw detection results
        for result in results:
            box = result['box']
            x, y, w, h = int(box['x']), int(box['y']), int(box['w']), int(box['h'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw mask if enabled
            if self.show_mask and 'mask' in result:
                mask = np.array(result['mask'])
                mask = (mask > 0.5).astype(np.uint8) * 255
                mask = cv2.resize(mask, (w, h))
                
                # Create colored overlay
                overlay = frame[y:y+h, x:x+w].copy()
                overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                frame[y:y+h, x:x+w] = overlay
            
            # Draw label with confidence
            label = f"{result.get('match', 'Unknown')} ({result['confidence']:.2f})"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw additional metadata if available
            if 'metadata' in result:
                metadata = result['metadata']
                y_offset = y + h + 20
                for key, value in metadata.items():
                    text = f"{key}: {value}"
                    cv2.putText(frame, text, (x, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_offset += 20
    
    def process_frame(self, frame):
        # Calculate FPS
        self.calculate_fps()
        
        # Process frame through pipeline
        results = self.pipeline.process_frame(frame)
        
        # Filter results by confidence threshold
        results = [r for r in results if r['confidence'] > self.confidence_threshold]
        
        # Draw results
        self.draw_results(frame, results)
        
        return frame
    
    def run(self):
        print("Starting real-time object detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Toggle mask display")
        print("  'f' - Toggle FPS display")
        print("  '+' - Increase confidence threshold")
        print("  '-' - Decrease confidence threshold")
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Show frame
            cv2.imshow('Object Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.show_mask = not self.show_mask
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            elif key == ord('+'):
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    app = ObjectDetectionApp()
    app.run()

if __name__ == '__main__':
    main() 