import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialStreamVGG(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # RGB input: 3 channels
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, rgb):
        """
        Args:
            rgb: [batch, 3, H, W] - RGB frames
        Returns:
            logits: [batch, num_classes]
        """
        x = self.features(rgb)
        logits = self.classifier(x)
        return logits


class TemporalStreamVGG(nn.Module):
    
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()
        
        # Optical flow input: 2 * num_frames channels (x and y direction)
        flow_channels = num_frames * 2
        
        self.features = nn.Sequential(
            nn.Conv2d(flow_channels, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, flow):
        """
        Args:
            flow: [batch, 2*num_frames, H, W] - Stacked optical flow
        Returns:
            logits: [batch, num_classes]
        """
        x = self.features(flow)
        logits = self.classifier(x)
        return logits