import torch as t
import torch.nn as nn

class EarlyFusion(nn.Module):
    def __init__(self, in_size=64, num_classes=10, num_frames=10):
        super().__init__()

        in_channels = 3 * num_frames  # 3 channels (RGB) per frame

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 128 → 64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 64 → 32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 32 → 16
        )
        
        flat = 64 * (in_size//8) * (in_size//8)            # 64*16*16
        self.classifier = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),                          # logits
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x = self._early_fusion(x)  # B, C*T, H, W
        B, CT, H, W = x.shape
        x = self.features(x)
        x = x.view(B, -1)
        x = self.classifier(x)
        return x

    def _early_fusion(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B, C*T, H, W) # B, C*T, H, W
        return x