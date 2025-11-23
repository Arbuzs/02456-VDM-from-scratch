import torch as t
import wandb
from utils.logger import get_logger
from config import settings
from typing import Dict, Any
from rich.progress import Progress
from metrics.VDM_metrics import FIDScore, IS, SNR, NLL, BPD


logger = get_logger(__name__) 

class Experiment:
   
    def __init__(
            self, 
            project_name: str,
            name: str,
            config: dict
            ):
        self.project_name = project_name
        self.name = name
        self.config = config

        # Evaluation metrics
        # Val
        self.fid_metric_val = FIDScore(device='cuda')
        self.is_metric_val = IS(device='cuda')
        self.snr_metric_val = SNR()
        self.nll_metric_val = NLL()
        
        # Evaluation metrics
        # Test
        self.fid_metric_test = FIDScore(device='cuda')
        self.is_metric_test = IS(device='cuda')
        self.snr_metric_test = SNR()
        self.nll_metric_test = NLL()
        
        
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
            total=len(self.config['val_loader'])
            )
        self.task_test = self.progress.add_task(
            "[yellow]Testing model...",
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
            for X in self.config['val_loader']:
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

    def test_evaluation(self):
        """
        Runs the VLB metric aggregation loop over the test set.
        """
        self.config['model'].eval()
        self.config['loss_function'].eval() # Also set loss_fn to eval mode

        total_metrics = {} # Use a dict to aggregate metrics
        num_batches = 0
        
        # Reset the progress task for a clean run
        self.progress.reset(self.task_test)

        with t.no_grad():
            # Use the test_loader instead of val_loader
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
                self.progress.update(self.task_test, advance=1)
        
        # Return the average of all metrics over the test set, prefixed with 'test/'
        avg_metrics = {f'test/{key}': value / num_batches for key, value in total_metrics.items()}
        return avg_metrics

        
    def final_evaluation(self):
        """
        Run comprehensive evaluation on test set after training.
        Computes FID, IS, SNR metrics on generated vs real images.
        """
        logger.info("Running final evaluation on test set...")
        self.config['model'].eval()
        
        # Get evaluation parameters from config
        n_generated_samples = self.config['n_eval_samples']
        n_sample_steps = self.config['n_sample_steps']
        eval_batch_size = self.config['eval_batch_size']
        
        # --- Collect real images from test set ---
        logger.info("Collecting real images from test set...")
        real_images_list = []
        
        with t.no_grad():
            for batch in self.config['test_loader']:
                if isinstance(batch, (list, tuple)):
                    X = batch[0]
                else:
                    X = batch
                
                # Normalize to [0, 1]
                if X.min() < 0:
                    X = (X + 1) / 2
                X = t.clamp(X, 0, 1)
                
                real_images_list.append(X.cpu())
        
        real_images = t.cat(real_images_list, dim=0)
        logger.info(f"Collected {real_images.shape[0]} real images")
        
        # --- Generate samples ---
        logger.info(f"Generating {n_generated_samples} samples...")
        generated_images_list = []
        
        with t.no_grad():
            for i in range(0, n_generated_samples, eval_batch_size):
                current_batch_size = min(eval_batch_size, n_generated_samples - i)
                samples = self.config['model'].sample(
                    batch_size=current_batch_size,
                    n_sample_steps=n_sample_steps,
                    clip_samples=True
                )
                
                # Normalize to [0, 1]
                if samples.min() < 0:
                    samples = (samples + 1) / 2
                samples = t.clamp(samples, 0, 1)
                
                generated_images_list.append(samples.cpu())
        
        generated_images = t.cat(generated_images_list, dim=0)
        logger.info(f"Generated {generated_images.shape[0]} samples")
        
        # --- Compute metrics ---
        logger.info("Computing evaluation metrics...")
        eval_results = {}
        
                # FID and IS
        logger.info("Computing FID...")
        for i in range(0, min(len(real_images), len(generated_images)), eval_batch_size):
            real_batch = real_images[i:i+eval_batch_size].to(settings.device)
            gen_batch = generated_images[i:i+eval_batch_size].to(settings.device)
              
            self.fid_metric_test.update(real_batch, gen_batch)

        logger.info("Computing Inception Score...")
        for i in range(0, len(generated_images), eval_batch_size):
            gen_batch = generated_images[i:i+eval_batch_size].to(settings.device)
            self.is_metric_test.update(gen_batch)
        
        logger.info("Computing SNR...")
        for i in range(0, min(len(real_images), len(generated_images)), eval_batch_size):
            gen_batch = generated_images[i:i+eval_batch_size].to(settings.device)
            real_batch = real_images[i:i+eval_batch_size].to(settings.device)
            self.snr_metric_test.update(gen_batch, real_batch)
        
        # Compute final scores
        fid_score = self.fid_metric_test.compute()
        is_mean= self.is_metric_test.compute()
        snr_mean, snr_std = self.snr_metric_test.compute()
        
        eval_results['test/fid'] = fid_score
        eval_results['test/is_mean'] = is_mean
        eval_results['test/snr_mean_db'] = snr_mean
        eval_results['test/snr_std_db'] = snr_std
        
      
        return eval_results

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
            
            # Run final evaluation after all epochs complete
            logger.info("Training complete. Running final evaluation...")
            # This computes FID, IS, SNR, BPD, NLL metrics
            final_eval_results_comprehensive = self.final_evaluation()
            
            # This computes the aggregated VLB metrics (NLL, BPD if included in loss fn)
            final_eval_results_vlb = self.test_evaluation() 
            
            # Combine and Log final evaluation results
            final_eval_results = final_eval_results_comprehensive | final_eval_results_vlb

            self.experiment.log(final_eval_results)
            logger.info("Final evaluation results logged to wandb")

        except Exception as e: 
            logger.error(f"Experiment run failed with error: {e}")
            raise e # Re-raise the exception to see the full traceback
        
        finally:
            self.progress.stop()
            self.experiment.finish()