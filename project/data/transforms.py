from torchvision import transforms as T

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
    ])

transformFlow = T.Compose([
    T.RandomResizedCrop(64, scale=(0.8, 1.0)),  # Random crop with resize
    T.RandomHorizontalFlip(p=0.5),               # 50% chance to flip
    T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),                                           # RGB jitter
    T.ToTensor()
])