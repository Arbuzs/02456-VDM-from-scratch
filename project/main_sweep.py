import os
import torch as t
import wandb
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
# 1. Define the Training Configuration (Used as a template)
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
    sample_interval: int
    
    # NOTE: Checkpointing flag needs to be here if used by UNetVDM
    use_checkpointing: bool = False
    
    def __init__(self,
                 embedding_dim: int = 128,
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
                 n_samples_to_log: int = 9,
                 sample_interval: int = 10,
                 use_checkpointing: bool = True):
        
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


# ---
# 2. Main Training Function (The entry point for WandB Sweep)
# ---

def train():
    # 1. Initialize WandB and ingest parameters from the sweep configuration
    # Note: wandb.init() will look for sweep parameters if running in an agent.
    run = wandb.init(
        project='VDM-from-scratch', 
        job_type="sweep_run"
    )
    sweep_config = wandb.config

    # Define dataset-specifics (fixed for CIFAR-10)
    project_name = run.project
    epochs = sweep_config.epochs # Use epoch value from sweep config (50)
    dataset = settings.root_dir.split('/')[-1]
    image_shape = (3, 32, 32) 
    
    # 2. Create the Configuration object using sweep parameters
    # We map the sweep_config dictionary to the TrainConfig __init__
    cfg = TrainConfig(
        embedding_dim = sweep_config.embedding_dim,
        n_blocks = sweep_config.n_blocks,
        n_attention_heads = sweep_config.n_attention_heads,
        dropout_prob = sweep_config.dropout_prob,
        norm_groups = TrainConfig().norm_groups, # Use default or sweep if defined
        input_channels = TrainConfig().input_channels,
        use_fourier_features = TrainConfig().use_fourier_features,
        attention_everywhere = sweep_config.attention_everywhere,
        batch_size = sweep_config.batch_size,
        noise_schedule = sweep_config.noise_schedule,
        gamma_min = TrainConfig().gamma_min,
        gamma_max = TrainConfig().gamma_max,
        antithetic_time_sampling = TrainConfig().antithetic_time_sampling,
        lr = sweep_config.lr,
        weight_decay = sweep_config.weight_decay,
        clip_grad_norm = TrainConfig().clip_grad_norm,
        n_sample_steps = TrainConfig().n_sample_steps,
        clip_samples = TrainConfig().clip_samples,
        n_samples_to_log = TrainConfig().n_samples_to_log,
        sample_interval = TrainConfig().sample_interval,
        use_checkpointing = TrainConfig().use_checkpointing
    )
    
    # Note on DataLoaders: Since DataLoaders are imported pre-made objects,
    # they cannot use the swept batch_size. If batch_size were swept, 
    # the loaders would need to be created dynamically here based on cfg.batch_size.

    # 3. Initialize Model, Loss, and Optimizer
    print("Setting up model components...")
    
    unet = UNetVDM(cfg)
    vdm_model = VDM(unet, cfg, image_shape)
    loss_function = VLB(vdm_model)

    optimizer = t.optim.Adam(
        vdm_model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    # 4. Setup and Run the Experiment
    print("Setting up experiment...")
    diffusion_experiment = Experiment(
        project_name=project_name,
        # Use a descriptive name based on hyperparameters
        name=f"E{cfg.embedding_dim}_B{cfg.n_blocks}_LR{cfg.lr:.1e}_D{cfg.dropout_prob}_Att{str(cfg.attention_everywhere)[0]}",
        config={
            # Use imported DataLoaders
            'train_loader': cifar_image_trainloader,
            'val_loader': cifar_image_valloader,
            'test_loader': cifar_image_testloader, 
            'model': vdm_model,
            'loss_function': loss_function, 
            'optimizer': optimizer,
            'epochs': epochs,
            'dataset': dataset,
            # Pass all configurations for logging
            **cfg.__dict__ 
        },
    )

    print("Starting experiment run...")
    diffusion_experiment.run()
    run.finish() # Cleanly finish the run after the experiment is done

# ---
# 5. Main Execution: Run Sweep or Single Run
# ---

if __name__ == "__main__":
    # If a sweep controller is active, it will call this script
    # and wandb.init() inside the train function will automatically capture the config.
    
    if os.environ.get('WANDB_SWEEP_ID'):
        # If run by a sweep agent, just call train()
        train()
    else:
        # For a local, single run, run the full setup manually (or create a new job)
        print("Running in manual mode. Note: You should use 'wandb agent' or 'python run_sweep.py' for HPO.")
        # Re-using the manual run setup for local testing simplicity:
        
        # NOTE: Manually instantiate TrainConfig based on your desired test run,
        # not the one based on sweep_config default values.
        cfg_manual = TrainConfig() 
        
        # The logic below is simplified manual run, assuming it was adapted from the original __main__ logic:
        image_shape = (3, 32, 32)
        project_name = 'VDM-from-scratch'
        epochs = 3
        dataset = settings.root_dir.split('/')[-1]

        unet = UNetVDM(cfg_manual)
        vdm_model = VDM(unet, cfg_manual, image_shape)
        loss_function = VLB(vdm_model)
        optimizer = t.optim.Adam(
            vdm_model.parameters(), 
            lr=cfg_manual.lr, 
            weight_decay=cfg_manual.weight_decay
        )
        
        diffusion_experiment = Experiment(
            project_name=project_name,
            name='VDM_CIFAR10_Manual_Test',
            config={
                'train_loader': cifar_image_trainloader,
                'val_loader': cifar_image_valloader,
                'test_loader': cifar_image_testloader,
                'model': vdm_model,
                'loss_function': loss_function, 
                'optimizer': optimizer,
                'epochs': epochs,
                'dataset': dataset,
                **cfg_manual.__dict__ 
            },
        )
        # This part requires wandb.init() to be called inside the pipeline
        # which it is.
        diffusion_experiment.run()
        print("Experiment finished.")