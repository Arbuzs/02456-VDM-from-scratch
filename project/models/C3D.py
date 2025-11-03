import torch as t
import torch.nn as nn

class C3D(nn.Module):
    def __init__(self, in_size=64, num_classes=10, num_frames=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.MaxPool3d(2),  # 128 → 64

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.MaxPool3d(2),  # 64 → 32

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.MaxPool3d(2),  # 32 → 16
        )
        
        flat = 64 * (in_size//8) * (in_size//8)            # 64*16*16
        self.classifier = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)                               # logits
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.features(x)
        x = x.view(B, -1)
        x = self.classifier(x)
        return x