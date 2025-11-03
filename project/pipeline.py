import torch as t
import wandb
from utils.logger import get_logger
from config import settings
from typing import Dict, Any
from rich.progress import Progress
from metrics.classification import Accuracy

logger = get_logger(__name__) 

class Experiment:
    """
    Required in config:
    """
    def __init__(
            self, 
            project_name: str,
            name: str,
            config: dict
            # Dict[
                # t.nn.Module, 
                # t.utils.data.DataLoader,
                # Any
                # ],
            ):
        self.project_name = project_name
        self.name = name
        self.config = config
        
        self.progress = Progress()
        # Initialize metrics
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

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
        total_loss = 0.0
        num_batches = 0
        self.train_accuracy.reset()  # Reset accuracy at start of epoch
        
        for X, y in self.config['train_loader']:
            X, y = X.to(settings.device), y.to(settings.device)
            self.config['optimizer'].zero_grad()
            logits = self.config['model'](X)
            loss = self.config['loss_function'](logits, y)
            loss.backward()
            self.config['optimizer'].step()
            
            # Update metrics
            self.train_accuracy.update(logits, y)
            total_loss += loss.item()
            num_batches += 1
            
            self.progress.update(self.task_train, advance=1)
        
        return {
            'loss/train': total_loss / num_batches,
            'accuracy/train': self.train_accuracy.compute()
        }

    def eval(self):
        self.config['model'].eval()
        total_loss = 0.0
        num_batches = 0
        self.val_accuracy.reset()  # Reset accuracy at start of validation
        
        with t.no_grad():
            for X, y in self.config['test_loader']:
                X, y = X.to(settings.device), y.to(settings.device)
                logits = self.config['model'](X)
                loss = self.config['loss_function'](logits, y)
                
                # Update metrics
                self.val_accuracy.update(logits, y)
                total_loss += loss.item()
                num_batches += 1
                
                self.progress.update(self.task_val, advance=1)
        
        return {
            'loss/validation': total_loss / num_batches,
            'accuracy/validation': self.val_accuracy.compute()
        }

    def run(self):
        logger.info("Initializing Weights & Biases run")
        self.experiment = wandb.init(
            entity = 'IDLCV',
            project = self.project_name,
            name=self.name,
            config = self.config
        )

        logger.info("Starting experiment")
        self.progress.start()
        
        logger.info("Moving model to GPU")
        self.config['model'].to(settings.device)

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_results = self.train()
                test_results = self.eval()

                self.experiment.log(train_results | test_results)

                self.progress.reset(self.task_train)
                self.progress.reset(self.task_val)
                self.progress.update(self.task, advance=1)

        except Exception as e: # TODO: 
            logger.error(f"Experiment run failed with error: {e}")
        
        finally:
            self.progress.stop()
            self.experiment.finish()