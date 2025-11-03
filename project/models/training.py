# train_late_fusion_clean.py
import os, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from datasets import FrameVideoDataset
from models import LateFusion  # your class
save_dir = "/zhome/99/f/223556/project_2/src"
# -------------------------
# Repro & device selection
# -------------------------
# def set_seed(s=1337):
#     random.seed(s)
#     torch.manual_seed(s)
#     torch.cuda.manual_seed_all(s)

# set_seed(1337)
torch.backends.cudnn.benchmark = True

device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
use_cuda = device.type == "cuda"

print(f"Using device: {device}")

# -------------
# Data
# -------------
#root_dir = "/dtu/datasets1/02516/ucf101_noleakage"
root_dir = "/zhome/99/f/223556/project_2/ufc10"
transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

train_ds = FrameVideoDataset(root_dir=root_dir, split="train", transform=transform, stack_frames=True)
val_ds   = FrameVideoDataset(root_dir=root_dir, split="val",   transform=transform, stack_frames=True)

pin_memory = use_cuda  # only benefits CUDA

train_loader = DataLoader(
    train_ds, batch_size=8, shuffle=True,
    pin_memory=pin_memory )

val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False, pin_memory=pin_memory)

# -----------------
# Model & training
# -----------------
num_classes = 10   # set your value (or infer from dataset if you expose it)
num_frames  = 10

model = LateFusion(num_frames=num_frames, num_classes=num_classes, dropout_rate=0.5, fusion="average_pooling").to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # optional

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, non_blocking=use_cuda and pin_memory)
        y = y.long().to(device, non_blocking=use_cuda and pin_memory)

        logits = model(x)                    # [B, num_classes]
        loss = criterion(logits, y)

        total_loss   += loss.item() * y.size(0)
        total_correct+= (logits.argmax(1) == y).sum().item()
        total        += y.size(0)
    return total_loss / max(1,total), total_correct / max(1,total)

def train_one_epoch(model, loader, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=use_cuda and pin_memory)
        y = y.long().to(device, non_blocking=use_cuda and pin_memory)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss   += loss.item() * y.size(0)
        total_correct+= (logits.argmax(1) == y).sum().item()
        total        += y.size(0)

        pbar.set_postfix(
            train_loss=total_loss / max(1,total),
            train_acc =total_correct / max(1,total)
        )
    return total_loss / max(1,total), total_correct / max(1,total)

# -----------
# Run train
# -----------
epochs = 50
best_val_acc, best_epoch = 0.0, 0
train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
ckpt_path = os.path.join(save_dir, "late_fusion_best_leakage.pt")

for epoch in range(1, epochs+1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    va_loss, va_acc = evaluate(model, val_loader)

    # scheduler.step()  # optional

    train_loss_hist.append(tr_loss); train_acc_hist.append(tr_acc)
    val_loss_hist.append(va_loss);   val_acc_hist.append(va_acc)

    if va_acc > best_val_acc:
        best_val_acc, best_epoch = va_acc, epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "num_classes": num_classes,
        }, ckpt_path)

    print(f"Epoch {epoch:02d}/{epochs} | "
          f"train {tr_loss:.4f}/{tr_acc:.3f} | "
          f"val {va_loss:.4f}/{va_acc:.3f} | "
          f"best {best_val_acc:.3f} @ {best_epoch:02d}")

print(f"Saved best checkpoint to: {ckpt_path}")


# # Save last checkpoint (replace with best model saving logic if desired)
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epoch': epochs,
#     'num_classes': num_classes,
# }, 
# '/Users/emilianotorres/DTU_Masters/semester_1/DL_in_CV/project_2/last_checkpoint.pt')
