import torch
import torch.nn as nn
import torch.nn.functional as F

class RecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(RecognitionModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Get embeddings
        embeddings = self.classifier[0](features)
        
        # Get class predictions
        logits = self.classifier[1:](embeddings)
        
        return {
            'embeddings': embeddings,
            'logits': logits
        } 