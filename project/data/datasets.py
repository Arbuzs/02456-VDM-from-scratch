import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets, transforms
from typing import Literal, Optional, Tuple, Callable, Dict, Any
import numpy as np
from PIL import Image


class CIFAR10(torch.utils.data.Dataset): 
    def __init__(self, split='train', root_dir: str = './data', seed: int = 42, transform=None):
        self.root_dir = root_dir
        self.seed = seed
        self.split = split
        self.transform = transform
        
        # Load original CIFAR-10 datasets
        original_train = datasets.CIFAR10(
            root=self.root_dir, train=True, download=True
        )
        original_test = datasets.CIFAR10(
            root=self.root_dir, train=False, download=True
        )
        
        # Combine all 60k images
        all_data = torch.utils.data.ConcatDataset([original_train, original_test])
        
        # Split into 48k/6k/6k with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        train_split, val_split, test_split = random_split(
            all_data, 
            [48000, 6000, 6000],
            generator=generator
        )
        
        # Select the appropriate split
        if split == 'train':
            self.data = train_split
        elif split == 'val':
            self.data = val_split
        elif split == 'test':
            self.data = test_split
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image from the selected split, ignore label
        image, _ = self.data[idx]
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image