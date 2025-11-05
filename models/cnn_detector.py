import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetector, self).__init__()
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 5 + num_classes, kernel_size=1)  # 5 for box coordinates + num_classes
        )
        
        # Loss functions
        self.box_loss = nn.SmoothL1Loss()
        self.class_loss = nn.CrossEntropyLoss()
        
    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): Images to be processed
            targets (list[dict]): Ground-truth boxes present in the image (optional)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        # Process each image in the batch
        detections = []
        losses = {}
        
        for i, image in enumerate(images):
            # Feature extraction
            x = self.pool(F.relu(self.conv1(image)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            
            # Detection head
            x = self.detection_head(x)
            
            # Reshape output to [num_boxes, 5 + num_classes]
            batch_size = x.size(0)
            x = x.view(batch_size, -1, 5 + self.num_classes)
            
            if self.training:
                # Calculate losses
                target = targets[i]
                boxes = target['boxes']
                labels = target['labels']
                
                # Split predictions into box coordinates and class scores
                pred_boxes = x[..., :4]
                pred_scores = x[..., 4:]
                
                # Calculate box regression loss
                box_loss = self.box_loss(pred_boxes, boxes)
                
                # Calculate classification loss
                class_loss = self.class_loss(pred_scores, labels)
                
                # Store losses
                losses[f'box_loss_{i}'] = box_loss
                losses[f'class_loss_{i}'] = class_loss
                
            detections.append(x)
            
        if self.training:
            return losses
            
        return detections
    
    def compute_loss(self, predictions, targets):
        """
        Compute the loss for a batch of predictions
        """
        box_loss = 0
        class_loss = 0
        
        for pred, target in zip(predictions, targets):
            # Get predictions
            pred_boxes = pred[..., :4]
            pred_scores = pred[..., 4:]
            
            # Get targets
            boxes = target['boxes']
            labels = target['labels']
            
            # Calculate losses
            box_loss += self.box_loss(pred_boxes, boxes)
            class_loss += self.class_loss(pred_scores, labels)
            
        return {
            'box_loss': box_loss / len(predictions),
            'class_loss': class_loss / len(predictions)
        } 