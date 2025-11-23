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
# 1. Define the Training Configuration
# ---
@dataclass
class TrainConfig:
    # Model Config
    embedding_dim: int
    n_blocks: int
    n_attention_heads: int
    dropout_prob: float
    norm_groups: int
    input_channels: int
    use_fourier_features: bool
    attention_everywhere: bool
    
    # Training Config
    batch_size: int
    noise_schedule: str 
    gamma_min: float
    gamma_max: float
    antithetic_time_sampling: bool
    lr: float
    weight_decay: float
    clip_grad_norm: bool

    # --- SAMPLING CONFIG ---
    n_sample_steps: int
    clip_samples: bool
    n_samples_to_log: int
    
    def __init__(self,
                 embedding_dim: int = 64,   
                 n_blocks: int = 4,         
                 n_attention_heads: int = 4,
                 dropout_prob: float = 0.1,
                 norm_groups: int = 32,
                 input_channels: int = 3, 
                 use_fourier_features: bool = True,
                 attention_everywhere: bool = False,
                 batch_size: int = 128,      
                 noise_schedule: str = 'learned_linear',
                 gamma_min: float = -13.3,
                 gamma_max: float = 5.0,
                 antithetic_time_sampling: bool = True,
                 lr: float = 2e-4,
                 weight_decay: float = 0.0,
                 clip_grad_norm: bool = False,
                 
                 # --- SAMPLING DEFAULTS ---
                 n_sample_steps: int = 100,
                 clip_samples: bool = True,
                 n_samples_to_log: int = 9):
        
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

        # --- SAMPLING ---
        self.n_sample_steps = n_sample_steps
        self.clip_samples = clip_samples
        self.n_samples_to_log = n_samples_to_log

# ---
# 2. Setup Experiment Parameters
# ---

cfg = TrainConfig()

# Define dataset-specifics
project_name = 'VDM-from-scratch'
epochs = 2
dataset = settings.root_dir.split('/')[-1]
image_shape = (3, 32, 32) # For CIFAR-10

# ---
# 3. Initialize Model, Loss, and Optimizer
# ---

# The U-Net (noise predictor)
unet = UNetVDM(cfg)

# The VDM (wrapper model)
vdm_model = VDM(unet, cfg, image_shape)

# The loss function is now a separate module
loss_function = VLB(vdm_model)

# The optimizer still optimizes the VDM model's parameters
optimizer = t.optim.Adam(
    vdm_model.parameters(), 
    lr=cfg.lr, 
    weight_decay=cfg.weight_decay
)

# ---
# 4. Setup and Run the Experiment
# ---

print("Setting up experiment...")
diffusion_experiment = Experiment(
    project_name=project_name,
    name='VDM_CIFAR10_Run',
    config={
        'train_loader': cifar_image_trainloader,
        'test_loader': cifar_image_testloader,
        'val_loader':cifar_image_valloader,
        'model': vdm_model,
        'loss_function': loss_function, 
        'optimizer': optimizer,
        'epochs': epochs,
        'dataset': dataset,
        'n_eval_samples': 20,     
        'eval_batch_size': 10,    
        'n_sample_steps': 100,  
        **cfg.__dict__ # Log all config parameters to wandb
    },
)

print("Starting experiment run...")
diffusion_experiment.run()
print("Experiment finished.")