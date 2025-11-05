import torch
import cv2
import numpy as np
from models.cnn_detector import ObjectDetector
from models.unet_segmenter import UNet
from models.recognition_model import RecognitionModel
import faiss
import json

class ObjectDetectionPipeline:
    def __init__(self, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize models
        self.detector = ObjectDetector(num_classes).to(device)
        self.segmenter = UNet(n_channels=3, n_classes=1).to(device)
        self.recognizer = RecognitionModel(num_classes).to(device)
        
        # Initialize FAISS index for vector search
        self.index = faiss.IndexFlatL2(512)  # 512 is the embedding dimension
        self.product_database = []
        
    def preprocess_image(self, image):
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Resize to model input size
        image = cv2.resize(image, (416, 416))
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)
    
    def detect_objects(self, image):
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Get detections
        with torch.no_grad():
            detections = self.detector(input_tensor)
        
        # Convert detections to boxes
        boxes = self._process_detections(detections[0])
        return boxes
    
    def segment_objects(self, image, boxes):
        masks = []
        for box in boxes:
            x, y, w, h = box[:4]
            # Crop image to box
            crop = image[int(y):int(y+h), int(x):int(x+w)]
            if crop.size == 0:
                continue
                
            # Preprocess crop
            crop_tensor = self.preprocess_image(crop)
            
            # Get mask
            with torch.no_grad():
                mask = self.segmenter(crop_tensor)
            
            # Resize mask to original crop size
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
            masks.append(mask[0, 0].cpu().numpy())
            
        return masks
    
    def recognize_objects(self, image, boxes):
        results = []
        for box in boxes:
            x, y, w, h = box[:4]
            # Crop image to box
            crop = image[int(y):int(y+h), int(x):int(x+w)]
            if crop.size == 0:
                continue
                
            # Preprocess crop
            crop_tensor = self.preprocess_image(crop)
            
            # Get recognition results
            with torch.no_grad():
                output = self.recognizer(crop_tensor)
            
            # Get embedding and class
            embedding = output['embeddings'][0].cpu().numpy()
            logits = output['logits'][0].cpu().numpy()
            class_id = np.argmax(logits)
            confidence = float(np.max(logits))
            
            results.append({
                'embedding': embedding.tolist(),
                'class_id': int(class_id),
                'confidence': confidence
            })
            
        return results
    
    def search_database(self, embedding):
        # Search for nearest neighbor
        distances, indices = self.index.search(embedding.reshape(1, -1), k=1)
        
        if indices[0][0] == -1:  # No match found
            return None
            
        # Get product info
        product = self.product_database[indices[0][0]]
        return {
            'match': product['name'],
            'confidence': float(1.0 / (1.0 + distances[0][0])),
            'metadata': product['metadata']
        }
    
    def add_to_database(self, embedding, name, metadata=None):
        # Add embedding to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Add product info to database
        self.product_database.append({
            'name': name,
            'metadata': metadata or {}
        })
    
    def _process_detections(self, detections):
        # Convert raw detections to boxes
        boxes = []
        for detection in detections:
            x, y, w, h, conf, *class_scores = detection
            if conf > 0.5:  # Confidence threshold
                class_id = np.argmax(class_scores)
                boxes.append([x, y, w, h, conf, class_id])
        return boxes
    
    def process_frame(self, frame):
        # Detect objects
        boxes = self.detect_objects(frame)
        
        # Segment objects
        masks = self.segment_objects(frame, boxes)
        
        # Recognize objects
        recognition_results = self.recognize_objects(frame, boxes)
        
        # Search database for matches
        final_results = []
        for box, mask, rec_result in zip(boxes, masks, recognition_results):
            match = self.search_database(np.array(rec_result['embedding']))
            
            result = {
                'box': {
                    'x': float(box[0]),
                    'y': float(box[1]),
                    'w': float(box[2]),
                    'h': float(box[3])
                },
                'confidence': float(rec_result['confidence']),
                'mask': mask.tolist()
            }
            
            if match:
                result.update(match)
            
            final_results.append(result)
        
        return final_results 