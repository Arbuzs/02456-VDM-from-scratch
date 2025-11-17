from torch.utils.data import DataLoader
from data.datasets import CIFAR10
from data.transforms import transform_train, transform_test, transform_VDM_train

batch_size = 128

# Create datasets
train_dataset = CIFAR10(split='train', transform=transform_VDM_train)
val_dataset = CIFAR10(split='val', transform=transform_test)
test_dataset = CIFAR10(split='test', transform=transform_test)

# DataLoaders
cifar_image_trainloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,
)

cifar_image_valloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    )

cifar_image_testloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)


#lil test to check the shapes and if data is loaded properly
if __name__ == '__main__':
    
    #Check dataset sizes
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Test dataloaders
    print(f"Train batches: {len(cifar_image_trainloader)}")
    print(f"Val batches: {len(cifar_image_valloader)}")
    print(f"Test batches: {len(cifar_image_testloader)}")
    
    # Check a batch
    images = next(iter(cifar_image_trainloader))
    print(f"\nBatch shape: {images.shape}")