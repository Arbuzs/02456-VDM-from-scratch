import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ignite import metrics
import numpy as np
from scipy.stats import entropy
from scipy import linalg
import torchvision.transforms as T




class BaseMetricModule:
    pass

INCEPTION_TRANSFORM = T.Compose([
    T.Resize(299, antialias=True)
])



# --- 1. The Feature Extractor Wrapper (from previous step) ---
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, mode="features"):
        super().__init__()
        self.mode = mode
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.model.eval()
        if self.mode == "features":
            self.model.fc = nn.Identity()

    def forward(self, x):
        # This wrapper assumes x is already 299x299, or we can safety-resize here too.
        if self.mode == "features":
            x = self.model(x)
            return torch.flatten(x, start_dim=1)
        elif self.mode == "logits":
            return self.model(x)

# --- 2. Corrected FID Class with Resizing ---
class FIDScore:
    def __init__(self, device="cuda"):
        self.device = device
        
        # Initialize Wrapper
        self.wrapper = InceptionV3FeatureExtractor(mode="features").to(device)
        
        # Initialize Ignite Metric
        self.metric = metrics.FID(
            num_features=2048, 
            feature_extractor=self.wrapper, 
            device=device
        )

    def reset(self):
        self.metric.reset()

    def update(self, real_images, fake_images):
        """
        Args:
            real_images: Tensor [B, 3, 32, 32]
            fake_images: Tensor [B, 3, 32, 32]
        """
        # FIX: Resize images to 299x299 using interpolation
        # mode='bilinear' is standard for images. align_corners=False is standard for Inception.
        real_299 = F.interpolate(real_images, size=(299, 299), mode='bilinear', align_corners=False)
        fake_299 = F.interpolate(fake_images, size=(299, 299), mode='bilinear', align_corners=False)
        
        self.metric.update((real_299, fake_299))

    def compute(self):
        return self.metric.compute()

# --- 3. Corrected Inception Score Class with Resizing ---
class InceptionScoreMetric:
    def __init__(self, device="cuda", n_split=10):
        self.device = device
        
        # Initialize Wrapper (Logits mode for IS)
        self.wrapper = InceptionV3FeatureExtractor(mode="logits").to(device)
        
        self.metric = metrics.InceptionScore(
            num_features=1000, 
            feature_extractor=self.wrapper, 
            device=device
        )

    def reset(self):
        self.metric.reset()

    def update(self, fake_images):
        """
        Args:
            fake_images: Tensor [B, 3, 32, 32]
        """
        # FIX: Resize images to 299x299
        fake_299 = F.interpolate(fake_images, size=(299, 299), mode='bilinear', align_corners=False)
        
        self.metric.update(fake_299)

    def compute(self):
        return self.metric.compute()

class SignalToNoiseRatio(BaseMetricModule):
    """
    Signal-to-Noise Ratio (SNR) in dB.
    Measures the ratio of signal power to noise power in generated images.
    Higher is better.
    """
    
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self):
        self.snr_values = []
    
    def update(self, generated_images: torch.Tensor, reference_images: torch.Tensor = None):
        """
        Args:
            generated_images: Tensor of generated images (B, C, H, W).
            reference_images: Optional tensor of reference images (B, C, H, W).
        """
        with torch.no_grad():
            # Signal power: Mean square of the reference (or generated) image pixel values
            if reference_images is not None:
                # Mode 1: SNR between generated and reference
                # Signal is the reference, Noise is the residual/error
                noise = generated_images - reference_images
                signal_power = torch.mean(reference_images ** 2, dim=(1, 2, 3)) # Mean power over C, H, W
            else:
                # Mode 2: SNR using standard deviation as noise estimate
                # Signal is the image itself, Noise is the deviation from the mean
                signal_power = torch.mean(generated_images ** 2, dim=(1, 2, 3)) # Mean power over C, H, W
                # Calculate the noise (zero-mean version of the signal)
                noise = generated_images - torch.mean(generated_images, dim=(1, 2, 3), keepdim=True)
            
            # Noise power: Mean square of the noise tensor
            noise_power = torch.mean(noise ** 2, dim=(1, 2, 3)) # Mean power over C, H, W
            
            # Compute SNR in dB: SNR_dB = 10 * log10(Signal_Power / Noise_Power)
            # Add epsilon to prevent division by zero
            snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))
            self.snr_values.append(snr_db.cpu().numpy())
    
    def compute(self):
        """
        Returns:
            mean_snr: Mean SNR across all batches (in dB)
            std_snr: Std deviation of SNR across all batches (in dB)
        """
        if not self.snr_values:
            return float('nan'), float('nan')
        
        all_snr = np.concatenate(self.snr_values)
        return np.mean(all_snr), np.std(all_snr)

class NegativeLogLikelihood(BaseMetricModule):
    """
    Negative Log-Likelihood (NLL). Lower is better.
    Input `log_probs` must be the log-likelihood summed over all non-batch dimensions.
    """
    
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self):
        self.nll_sum = 0.0
        self.num_samples = 0
    
    def update(self, log_probs: torch.Tensor):
        """
        Args:
            log_probs: Log probabilities/likelihood per sample (B,). 
                       (Should not be the mean, but the sum per sample.)
        """
        with torch.no_grad():
            # Calculate negative log-likelihood for the batch
            # Note: The original code used -log_probs.mean(). If log_probs is already
            # the sum per sample, the metric should accumulate the sum of -log_probs.
            # Assuming 'log_probs' is the sum of log-likelihoods over C*H*W for each sample (B,):
            nll_batch_sum = -log_probs.sum().item() 
            batch_size = log_probs.shape[0]

            self.nll_sum += nll_batch_sum
            self.num_samples += batch_size
    
    def compute(self):
        """
        Returns:
            mean_nll: Average negative log-likelihood (NLL)
        """
        if self.num_samples == 0:
            return float('nan')
        
        return self.nll_sum / self.num_samples


class BitsPerDimension(BaseMetricModule):
    """
    Bits Per Dimension (BPD). Lower is better.
    A normalized version of NLL, useful for comparing models on different image sizes.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Total dimensionality of data (e.g., C * H * W for images).
        """
        super().__init__()
        if dim <= 0:
            raise ValueError("Dimension (dim) must be a positive integer (C * H * W).")
        # Conversion factor from nats (base e) to bits (base 2)
        self.nats_to_bits_factor = 1.0 / np.log(2) 
        self.dim = dim
        self.reset()
    
    def reset(self):
        self.bpd_sum = 0.0
        self.num_samples = 0
    
    def update(self, nll_per_sample: torch.Tensor):
        """
        Args:
            nll_per_sample: Negative log-likelihood per sample (B,). 
                            Must be calculated as -log_likelihood_sum_over_dims.
        """
        with torch.no_grad():
            # 1. Convert NLL from nats (base e) to bits
            nll_in_bits = nll_per_sample * self.nats_to_bits_factor
            
            # 2. Normalize by the total dimension
            bpd = nll_in_bits / self.dim
            
            self.bpd_sum += bpd.sum().item()
            self.num_samples += nll_per_sample.shape[0]
    
    def compute(self):
        """
        Returns:
            mean_bpd: Average bits per dimension
        """
        if self.num_samples == 0:
            return float('nan')
        
        return self.bpd_sum / self.num_samples

# Aliases for cleaner imports
FIDScore = FIDScore
IS = InceptionScoreMetric
SNR = SignalToNoiseRatio
NLL = NegativeLogLikelihood
BPD = BitsPerDimension

__all__ = [
    'FID', 'IS', 'SNR', 'NLL', 'BPD',
    'FIDScore', 'InceptionScoreMetric', 'SignalToNoiseRatio',
    'NegativeLogLikelihood', 'BitsPerDimension',
]