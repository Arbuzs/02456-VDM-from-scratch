import torch as t
import wandb
from utils.logger import get_logger
from config import settings
from rich.progress import Progress

logger = get_logger(__name__) 

class Experiment:
    """
    An experiment class adapted for VDM training.
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
        
        # Main epoch task
        self.task = self.progress.add_task(
            f"[red]Running {self.config['epochs']} epochs...",
            total=self.config['epochs']
            )

        # Training task (will be reset each epoch)
        self.task_train = self.progress.add_task(
            "[green]Training epoch...",
            total=len(self.config['train_loader'])
            )
        
        # Validation task (optional for VDM, usually involves sampling)
        if 'test_loader' in self.config and self.config['test_loader'] is not None:
             self.task_val = self.progress.add_task(
                "[blue]Validating epoch...",
                total=len(self.config['test_loader'])
                )
        else:
            self.task_val = None

    def train(self):
        self.config['model'].train()
        total_loss = 0.0
        num_batches = 0
        
        for images, _ in self.config['train_loader']:
            # 1. Move clean images to device
            x_start = images.to(settings.device)
            
            # 2. Sample random time steps for each image in the batch
            #    Range is [0, T-1] where T is total timesteps from model config
            t_steps = t.randint(0, self.config['model'].T, (x_start.shape[0],), device=settings.device).long()

            self.config['optimizer'].zero_grad()
            
            # 3. Calculate VDM loss (The VDM model handles the forward diffusion internally)
            #    We pass the 'criterion' (MSELoss) to the model's loss function.
            loss = self.config['model'].p_losses(x_start, t_steps, criterion=self.config['loss_function'])
            
            loss.backward()
            
            # Optional: Gradient clipping for stability
            t.nn.utils.clip_grad_norm_(self.config['model'].parameters(), 1.0)
            
            self.config['optimizer'].step()
            
            total_loss += loss.item()
            num_batches += 1
            
            self.progress.update(self.task_train, advance=1)
        
        return {
            'loss/train': total_loss / num_batches,
        }

    def eval(self):
        # VDM evaluation is tricky. Standard 'val loss' on noise prediction 
        # isn't always a great proxy for sample quality.
        # For now, we'll just compute the same noise MSE on the test set.
        if self.task_val is None: return {}

        self.config['model'].eval()
        total_loss = 0.0
        num_batches = 0
        
        with t.no_grad():
            for images, _ in self.config['test_loader']:
                x_start = images.to(settings.device)
                t_steps = t.randint(0, self.config['model'].T, (x_start.shape[0],), device=settings.device).long()
                
                loss = self.config['model'].p_losses(x_start, t_steps, criterion=self.config['loss_function'])
                total_loss += loss.item()
                num_batches += 1
                self.progress.update(self.task_val, advance=1)
        
        return {
            'loss/validation': total_loss / num_batches,
        }

    def run(self):
        logger.info("Initializing Weights & Biases run")
        # Filter config to only log serializable items to wandb if needed
        wandb_config = {k: v for k, v in self.config.items() if isinstance(v, (int, float, str, bool))}

        self.experiment = wandb.init(
            project=self.project_name,
            name=self.name,
            config=wandb_config
        )

        logger.info("Starting experiment")
        self.progress.start()
        
        # Ensure model is on the correct device
        self.config['model'].to(settings.device)

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_results = self.train()
                test_results = self.eval()

                # Log results to wandb
                self.experiment.log({**train_results, **test_results, 'epoch': epoch})

                # Reset progress bars for next epoch
                self.progress.reset(self.task_train)
                if self.task_val:
                    self.progress.reset(self.task_val)
                self.progress.update(self.task, advance=1)

            # TODO: Add model checkpoint saving here at the end of training

        except Exception as e:
            logger.error(f"Experiment run failed with error: {e}", exc_info=True)
            raise e
        
        finally:
            self.progress.stop()
            self.experiment.finish()