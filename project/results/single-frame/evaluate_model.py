import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
import pandas as pd

from model import SingleFrameCNN
from datasets import FrameImageDataset

# --- Configuration ---
root_dir = '/zhome/2d/5/168631/Documents/code/DLCV/Project 2/ufc10'
checkpoint_path = 'checkpoints/best_model.pth'

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

classes_list = ['Body-WeightSquats', 'HandstandPushups', 'HandstandWalking', 'JumpingJack', 'JumpRope',
                'Lunges', 'PullUps', 'PushUps', 'TrampolineJumping', 'WallPushups']
num_classes = len(classes_list)

# --- Data Loading ---
test_dataset = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# --- Model Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SingleFrameCNN(num_classes).to(device)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Model had val_acc: {checkpoint['val_acc']:.2f}%")

# --- Test ---
model.eval()
correct = 0
total = 0
class_correct = [0.] * num_classes
class_total = [0.] * num_classes

with torch.no_grad():
    for frames, labels in test_loader:
        frames = frames.to(device)
        labels = labels.to(device)
        
        outputs = model(frames)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

test_acc = 100. * correct / total
class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(num_classes)]

print(f"\nOverall Test Accuracy: {test_acc:.2f}%\n")
print("Per-class Test Accuracy:")
print("-" * 50)
for i, class_name in enumerate(classes_list):
    print(f"{class_name:25s}: {class_acc[i]:6.2f}%")

# Save results
df_results = pd.DataFrame({'class': classes_list + ['Overall'], 'test_accuracy': class_acc + [test_acc]})
df_results.to_csv('test_results.csv', index=False)
print("\nTest results saved to: test_results.csv")