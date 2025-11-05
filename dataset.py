import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
from torchvision import transforms
from PIL import Image

class PhoneDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotation_file (string): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations
        with open(os.path.join(root_dir, annotation_file), 'r') as f:
            self.annotations = json.load(f)
            
        # Create image_id to annotations mapping
        self.image_annotations = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in self.image_annotations:
                self.image_annotations[ann['image_id']] = []
            self.image_annotations[ann['image_id']].append(ann)
            
        # Create image_id to image mapping
        self.images = {img['id']: img for img in self.annotations['images']}
        
        # Verify all images exist
        for img in self.images.values():
            img_path = os.path.join(self.root_dir, img['file_name'])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        image_id = list(self.images.keys())[idx]
        image_info = self.images[image_id]
        
        # Load image
        img_path = os.path.join(self.root_dir, image_info['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise
        
        # Get annotations for this image
        annotations = self.image_annotations.get(image_id, [])
        
        # Prepare target
        boxes = []
        labels = []
        
        for ann in annotations:
            # Convert bbox from [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }
        
        return image, target

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized bounding boxes
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets 