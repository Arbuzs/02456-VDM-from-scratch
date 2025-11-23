import os
import torch as t
import wandb
from utils.logger import get_logger
from config import settings
from typing import Dict, Any
from rich.progress import Progress
import matplotlib.pyplot as plt
# from metrics.classification import Accuracy # No longer needed
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
        
        # For tracking the best model
        self.best_val_bpd = float('inf')

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
        
        if settings.device == 'cuda':
            t.cuda.empty_cache()
        
        # Return the average of all metrics over the epoch
        avg_metrics = {f'val/{key}': value / num_batches for key, value in total_metrics.items()}
        return avg_metrics

    # SAMPLING FUNCTION (From pure noise)
    def sample_and_log_images(self, epoch):
        logger.info(f"Sampling images for epoch {epoch}...")
        model = self.config['model']
        model.eval() # Set model to evaluation mode

        # Get sampling parameters from config
        n_samples = self.config.get('n_samples_to_log', 3)
        n_steps = self.config.get('n_sample_steps', 100)
        clip = self.config.get('clip_samples', True)
        
        with t.no_grad():
            sample_images = model.sample(
                batch_size=n_samples, 
                n_sample_steps=n_steps, 
                clip_samples=clip
            )
            sample_images = sample_images.cpu() 
        
        if settings.device == 'cuda':
            t.cuda.empty_cache()

        # Return a dictionary formatted for wandb.log()
        return {
            "epoch": epoch,
            "generated_images": [wandb.Image(img) for img in sample_images]
        }

    # SAMPLING FUNCTION (From real images)
    def reconstruct_and_log_images(self, epoch):
        logger.info(f"Reconstructing images for epoch {epoch}...")
        model = self.config['model']
        model.eval()
        
        # Get batch of real images from validation set
        # We just grab the first batch in the iterator
        val_loader = self.config['val_loader']
        x_real_batch = next(iter(val_loader))
        x_real_batch = x_real_batch.to(settings.device)

        n_samples = self.config.get('n_samples_to_log', 3)
        # Take the first N images from the batch validation set
        x_real = x_real_batch[:n_samples]

        n_steps = self.config.get('n_sample_steps', 100)
        clip = self.config.get('clip_samples', True)
        
        with t.no_grad():
            # Call the new reconstruct method
            # x_noisy will be the image at t=1.0 (pure noise)
            # x_recon will be the image after denoising
            x_noisy, x_recon = model.reconstruct(
                x=x_real,
                n_sample_steps=n_steps,
                clip_samples=clip,
                t_start=1.0 # Full reconstruction from pure noise
            )
            
            x_real = x_real.cpu()
            x_noisy = x_noisy.cpu()
            x_recon = x_recon.cpu()

        # Combine images side-by-side for easier comparison
        # Format: [Real Image] [Noisy Latent] [Reconstruction]
        # We concatenate along the width dimension (dim=2 for C,H,W or dim=3 for B,C,H,W)
        combined_images = t.cat([x_real, x_noisy, x_recon], dim=3)

        return {
            "epoch": epoch,
            "reconstructions": [wandb.Image(img, caption="Real | Noisy | Recon") for img in combined_images]
        }

    def log_noise_schedule(self, epoch):
        model = self.config['model']
        model.eval()
        
        # Create time axis from 0 to 1
        # Using 100 points to make it smooth (in case of using non-linear later on)
        t_vals = t.linspace(0, 1, 100, device=settings.device)
        
        # Get the Log-SNR (gamma) from the model
        with t.no_grad():
            gamma_vals = model.gamma(t_vals)
            
        # Move to CPU for plotting
        t_cpu = t_vals.cpu().numpy()
        gamma_cpu = -gamma_vals.cpu().numpy()
        
        # Create the Plot using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_cpu, gamma_cpu, label=f"Epoch {epoch}")
        ax.set_title("Learned Noise Schedule (Log-SNR)")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Log-SNR $\gamma(t)$")
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Convert to WandB Image
        plot_image = wandb.Image(fig)
        
        # Close the plot to save memory
        plt.close(fig)
        
        return {"noise_schedule_plot": plot_image}

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Saves model and optimizer state. 
        Saves 'last.pth' every call, and 'best.pth' if is_best=True.
        """
        # Create directory: checkpoints/{Run_Name}
        ckpt_dir = f"project/results/checkpoints/{self.name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Prepare state dict
        # We assume 'model' and 'optimizer' are in config
        state = {
            'epoch': epoch,
            'model_state_dict': self.config['model'].state_dict(),
            'optimizer_state_dict': self.config['optimizer'].state_dict(),
            'metrics': metrics,
        }
        
        # Save the current latest checkpoint
        last_path = os.path.join(ckpt_dir, "last.pth")
        t.save(state, last_path)
        
        # Save the checkpoint with best BPD
        if is_best:
            best_path = os.path.join(ckpt_dir, "best.pth")
            t.save(state, best_path)
            logger.info(f"Saved new best model (BPD: {metrics.get('val/bpd', 'N/A'):.4f})")

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

        # Get the sample interval
        sample_interval = self.config.get('sample_interval', 10)

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_results = self.train()
                val_results = self.eval()
                
                # --- LOGGING ---
                # Create a single dict to log
                log_data = train_results | val_results

                # Saving model checkpoint
                current_bpd = val_results.get('val/bpd', float('inf'))
                is_best = False
                if current_bpd < self.best_val_bpd:
                    self.best_val_bpd = current_bpd
                    is_best = True

                # Overwrites the last.pth with the current model and updates best.pth if we improved bpd
                self.save_checkpoint(epoch, log_data, is_best=is_best)

                is_first = (epoch == 1)
                is_interval = (epoch % sample_interval == 0)
                is_last = (epoch == self.config['epochs'])
                
                should_sample = is_first or is_interval or is_last

                if should_sample:
                    # Log the log SNR against t plot
                    schedule_log = self.log_noise_schedule(epoch)
                    log_data.update(schedule_log)
                    
                    # Generate images from noise and add to the log data
                    gen_image_log = self.sample_and_log_images(epoch)
                    log_data.update(gen_image_log)

                    # Reconstruct images and add to the log data
                    recon_image_log = self.reconstruct_and_log_images(epoch)
                    log_data.update(recon_image_log)
                
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