from torch.utils.data import DataLoader
from data.datasets import get_cifar10_image_dataset
from config import settings

cifar_image_trainset = get_cifar10_image_dataset(
    root_dir=settings.root_dir, 
    split='train'
)

cifar_image_testset = get_cifar10_image_dataset(
    root_dir=settings.root_dir, 
    split='val' # CIFAR10 'val' just means 'test' set
)

cifar_image_trainloader = DataLoader(
    cifar_image_trainset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

cifar_image_testloader = DataLoader(
    cifar_image_testset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)