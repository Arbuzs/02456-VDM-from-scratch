import os
import sys
import torch as t
from dataclasses import dataclass

from pipeline import Experiment
from data.dataloaders import (
    cifar_image_trainloader,
    cifar_image_testloader,
    cifar_image_valloader
)
from models.vdm_unet import UNetVDM
from models.vdm_no_loss import VDM
from losses.VLB import VLB 
from config import settings

# ---
# 1. Configuration (Must Match Training Run)
# ---
# We re-create the exact configuration class used for training.
@dataclass
class TrainConfig:
    embedding_dim: int
    n_blocks: int
    n_attention_heads: int
    dropout_prob: float
    norm_groups: int
    input_channels: int
    use_fourier_features: bool
    attention_everywhere: bool
    batch_size: int
    noise_schedule: str 
    gamma_min: float
    gamma_max: float
    antithetic_time_sampling: bool
    lr: float
    weight_decay: float
    clip_grad_norm: bool
    n_sample_steps: int
    clip_samples: bool
    n_samples_to_log: int
    sample_interval: int
    use_checkpointing: bool
    
    def __init__(self,
                 embedding_dim: int = 128,   
                 n_blocks: int = 4,         
                 n_attention_heads: int = 4,
                 dropout_prob: float = 0.1,
                 norm_groups: int = 32,
                 input_channels: int = 3, 
                 use_fourier_features: bool = True,
                 attention_everywhere: bool = False, # Set to False based on discussion
                 batch_size: int = 128,      
                 noise_schedule: str = 'learned_linear',
                 gamma_min: float = -13.3,
                 gamma_max: float = 5.0,
                 antithetic_time_sampling: bool = True,
                 lr: float = 2e-4,
                 weight_decay: float = 0.0,
                 clip_grad_norm: bool = False,
                 n_sample_steps: int = 100,
                 clip_samples: bool = True,
                 n_samples_to_log: int = 9,
                 sample_interval: int = 10,
                 use_checkpointing: bool = True): 
        
        # ... (Parameter assignments omitted for brevity, but they are necessary) ...
        self.embedding_dim = embedding_dim
        self.n_blocks = n_blocks
        self.n_attention_heads = n_attention_heads
        self.dropout_prob = dropout_prob
        self.norm_groups = norm_groups
        self.input_channels = input_channels
        self.use_fourier_features = use_fourier_features
        self.attention_everywhere = attention_everywhere
        self.batch_size = batch_size
        self.noise_schedule = noise_schedule
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.antithetic_time_sampling = antithetic_time_sampling
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.n_sample_steps = n_sample_steps
        self.clip_samples = clip_samples
        self.n_samples_to_log = n_samples_to_log
        self.sample_interval = sample_interval
        self.use_checkpointing = use_checkpointing


# --- USER CONFIGURATION ---
# IMPORTANT: Change this to the exact path of your best checkpoint file.
CHECKPOINT_PATH = "project/results/checkpoints/VDM_CIFAR10_Run/best.pth"
RUN_NAME = "VDM_CIFAR10_Best_Test"

# --- 2. Initialize Model and Load State ---

# Initialize
cfg = TrainConfig() 
image_shape = (3, 32, 32)
device = settings.device

# Initialize the model structure
unet = UNetVDM(cfg)
vdm_model = VDM(unet, cfg, image_shape)
loss_function = VLB(vdm_model) # Initialize VLB so we can use its logic

# Load Checkpoint
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
    sys.exit(1)

print(f"Loading model state from {CHECKPOINT_PATH}...")
checkpoint = t.load(CHECKPOINT_PATH, map_location=device)

# Load model state (the model must be on the CPU/GPU it was saved on, 
# or you must use map_location)
vdm_model.load_state_dict(checkpoint['model_state_dict'])

# Move model to device
vdm_model.to(device)
vdm_model.eval()
loss_function.to(device)
dataset = settings.root_dir.split('/')[-1]

optimizer = t.optim.Adam(
    vdm_model.parameters(), 
    lr=cfg.lr, 
    weight_decay=cfg.weight_decay
)


# --- 3. Setup Test Data and Experiment ---
project_name = 'VDM-from-scratch'

# Initialize the Experiment class 
# We pass the loaded components into the config
best_checkpoint = Experiment(
    project_name=project_name,
    name='VDM_CIFAR10_Run_best_checkpoint',
    config={
        'train_loader': cifar_image_trainloader,
        'test_loader': cifar_image_testloader,
        'val_loader':cifar_image_valloader,
        'model': vdm_model,
        'loss_function': loss_function, 
        'optimizer': optimizer,
        'epochs': 1,
        'dataset': dataset,
        'n_eval_samples': 200,     
        'eval_batch_size': 20,    
        'n_sample_steps': 100,
        **cfg.__dict__ # Log all config parameters to wandb
    },
)

# --- 4. Run Final Evaluation ---

print("Starting best checkpoint run...")
# We only call the run method to initialize wandb and then call test()
best_checkpoint.run()
