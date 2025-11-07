import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def get_cifar10_image_dataset(root_dir='./data', split='train'):
    """
    Standard CIFAR10 dataset returning [C, H, W] images.
    """
    is_train = (split == 'train')
        
    dataset = datasets.CIFAR10(
        root=root_dir,
        train=is_train,
        download=True, # Automatically downloads if not present
        transform=transforms.Compose([transforms.ToTensor()])
    )
    return dataset

# --- Usage Example ---
if __name__ == '__main__':
    # Standard Image Loader    
    cifar_image_dataset = get_cifar10_image_dataset(split='train')
    cifar_image_loader = DataLoader(cifar_image_dataset, batch_size=8, shuffle=True)
    
    images, labels = next(iter(cifar_image_loader))
    print(f"Standard Image Shape: {images.shape}") 
    # Output: [8, 3, 64, 64] (Batch, Channels, Height, Width)