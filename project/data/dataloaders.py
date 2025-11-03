from torch.utils.data import DataLoader
from data.datasets import FrameVideoDataset, FrameImageDataset,FrameFlowDataset
from data.transforms import transform
from config import settings

framevideostack_trainset = FrameVideoDataset(
    root_dir=settings.root_dir,
    split='train',
    transform=transform,
    stack_frames = True
    )

framevideostack_testset = FrameVideoDataset(
    root_dir=settings.root_dir,
    split='val',
    transform=transform,
    stack_frames = True
    )

framevideostack_trainloader = DataLoader(
    framevideostack_trainset,
    batch_size=32,
    shuffle=True,
    num_workers=4
    )

framevideostack_testloader = DataLoader(
    framevideostack_testset,
    batch_size=32,
    shuffle=False,
    num_workers=4
    )

frameimage_trainset = FrameImageDataset(
    root_dir=settings.root_dir,
    split='train',
    transform=transform
    )

frameimage_testset = FrameImageDataset(
    root_dir=settings.root_dir,
    split='val',
    transform=transform
    )

frameimage_trainloader = DataLoader(
    frameimage_trainset, 
    batch_size=32,
    shuffle=True,
    num_workers=4
    )

frameimage_testloader = DataLoader(
    frameimage_testset,
    batch_size=32,
    shuffle=False,
    num_workers=4)



frameimage_trainset = FrameFlowDataset(
    root_dir=settings.root_dir,
    split='train',
    transform=None
    )

frameimage_valset = FrameFlowDataset(
    root_dir=settings.root_dir,
    split='val',
    transform=None
    )

frameimage_testset = FrameFlowDataset(
    root_dir=settings.root_dir,
    split='test',
    transform=None
    )


frameflow_trainloader = DataLoader(
    frameimage_trainset, 
    batch_size=32,
    shuffle=True,
    num_workers=4
    )

frameflow_valloader = DataLoader(
    frameimage_testset,
    batch_size=32,
    shuffle=False,
    num_workers=4)

frameflow_testloader = DataLoader(
    frameimage_testset,
    batch_size=32,
    shuffle=False,
    num_workers=4)