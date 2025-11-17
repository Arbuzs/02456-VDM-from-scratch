from torchvision import transforms as T
import torch
import random



transform_train = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])

transform_test = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])


transform_VDM_train = T.Compose([
    T.Resize((32, 32)),
    T.RandomHorizontalFlip(p=0.5),  
    T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5), 
    T.ToTensor(),
 
])

