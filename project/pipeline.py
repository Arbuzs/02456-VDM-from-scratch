import torch as t
import wandb
from utils.logger import get_logger
from config import settings
from typing import Dict, Any
from rich.progress import Progress
# from metrics.classification import Accuracy # No longer needed

logger = get_logger(__name__) 

class Experiment:
    """
    Required in config:
        'model': The VDM model
        'loss_function': The VLB loss module
        'train_loader': Training dataloader
        'test_loader': Validation/Test dataloader
        'optimizer': The optimizer
        'epochs': Number of epochs
    """
    def __init__(
            self, 
            project_name: str,
            name: str,
            config: dict
            ):
        self.project_name = project_name
        self.name = name
        self.config = config
        
        self.progress = Progress()
        
        self.task = self.progress.add_task(
            f"[red]Running {self.config['epochs']} epochs...",
            total=self.config['epochs']
            )

        self.task_train = self.progress.add_task(
            "[green]Training epoch...",
            total=len(self.config['train_loader'])
            )
        
        self.task_val = self.progress.add_task(
            "[blue]Validating epoch...",
            total=len(self.config['test_loader'])
            )
        
    def _parse_config(self):
        return {k:f'{v=}'.split('=')[0] for k, v in self.config.items()}

    def train(self):
        self.config['model'].train()
        self.config['loss_function'].train() # Also set loss_fn to train mode
        
        total_metrics = {} # Use a dict to aggregate metrics from the model
        num_batches = 0
        
        for X in self.config['train_loader']:
            X = X.to(settings.device)
            batch = (X, None) # Dataloader only yields X
            
            self.config['optimizer'].zero_grad()
            
            # 1. Model forward pass
            model_out, batch_data = self.config['model'](batch)
            
            # 2. Loss function computes the VLB
            loss, metrics = self.config['loss_function'](model_out, batch_data)

            loss.backward()
            self.config['optimizer'].step()
            
            # Aggregate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value.item() if isinstance(value, t.Tensor) else value
            
            num_batches += 1
            self.progress.update(self.task_train, advance=1)
        
        # Return the average of all metrics over the epoch
        avg_metrics = {f'train/{key}': value / num_batches for key, value in total_metrics.items()}
        return avg_metrics

    def eval(self):
        self.config['model'].eval()
        self.config['loss_function'].eval() # Also set loss_fn to eval mode

        total_metrics = {} # Use a dict to aggregate metrics
        num_batches = 0
        
        with t.no_grad():
            for X in self.config['test_loader']:
                X = X.to(settings.device)
                batch = (X, None) # Dataloader only yields X

                # 1. Model forward pass
                model_out, batch_data = self.config['model'](batch)
                
                # 2. Loss function computes the VLB
                loss, metrics = self.config['loss_function'](model_out, batch_data)
                
                # Aggregate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value.item() if isinstance(value, t.Tensor) else value
                
                num_batches += 1
                self.progress.update(self.task_val, advance=1)
        
        # Return the average of all metrics over the epoch
        avg_metrics = {f'val/{key}': value / num_batches for key, value in total_metrics.items()}
        return avg_metrics

    # --- SAMPLING FUNCTION ---
    def sample_and_log_images(self, epoch):
        logger.info(f"Sampling images for epoch {epoch}...")
        model = self.config['model']
        model.eval() # Set model to evaluation mode

        # Get sampling parameters from config
        n_samples = self.config.get('n_samples_to_log', 9)
        n_steps = self.config.get('n_sample_steps', 100)
        clip = self.config.get('clip_samples', True)
        
        with t.no_grad():
            sample_images = model.sample(
                batch_size=n_samples, 
                n_sample_steps=n_steps, 
                clip_samples=clip
            )
            # .cpu() is good practice before logging
            sample_images = sample_images.cpu() 

        # Return a dictionary formatted for wandb.log()
        return {
            "epoch": epoch,
            "sampled_images": [wandb.Image(img) for img in sample_images]
        }

    def run(self):
        logger.info("Initializing Weights & Biases run")
        self.experiment = wandb.init(
            project = self.project_name,
            name=self.name,
            config = self.config
        )
        # Watch the model
        self.experiment.watch(self.config['model'], log="all", log_freq=100)

        logger.info("Starting experiment")
        self.progress.start()
        
        if settings.device == 'cuda':
            logger.info("Clearing CUDA cache...")
            t.cuda.empty_cache()
            
        logger.info("Moving model and loss function to GPU")
        self.config['model'].to(settings.device)
        self.config['loss_function'].to(settings.device)

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_results = self.train()
                test_results = self.eval()
                
                # --- LOGGING ---
                # Create a single dict to log
                log_data = train_results | test_results
                
                # Sample and add images to the log data
                image_log = self.sample_and_log_images(epoch)
                log_data.update(image_log)
                
                # Log everything for this epoch
                self.experiment.log(log_data)

                self.progress.reset(self.task_train)
                self.progress.reset(self.task_val)
                self.progress.update(self.task, advance=1)

        except Exception as e: 
            logger.error(f"Experiment run failed with error: {e}")
            raise e # Re-raise the exception to see the full traceback
        
        finally:
            self.progress.stop()
            self.experiment.finish()