import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleFrameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.convolutional = nn.Sequential(

                nn.Conv2d(3,16,3,padding=1), 
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16,32,3,padding=1), #64->64
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.MaxPool2d(2,2), #64 - > 32 

                nn.Conv2d(32,32,3,padding=1), #32 ->32
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                
                nn.Conv2d(32,64,3,padding=1), #32 ->32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.MaxPool2d(2,2), #32 ->16

                nn.Conv2d(64,128,3,padding=1), #16 ->16
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                
                nn.Conv2d(128,128,3,padding=1), #16 ->16
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.MaxPool2d(2,2), #16 -> 8


                nn.Conv2d(128,128,3,padding=1), #8 ->8
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                
                nn.Conv2d(128,128,3,padding=1), #8 ->8
                nn.BatchNorm2d(128),
                nn.ReLU(),
            

                nn.MaxPool2d(2,2), #8 -> 4
                nn.Flatten(),
                
                nn.Linear(4*4*128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            
                nn.Linear(64,num_classes)
                )

    def forward(self, x):
            
            return self.convolutional(x)

  