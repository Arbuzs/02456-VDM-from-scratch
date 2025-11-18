import numpy as np
import torch
from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
from torch.special import expm1
from tqdm import trange

# Helper functions
def maybe_unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch
    else:
        return batch, None

def unsqueeze_right(x, num_dims=1):
    """Unsqueezes the last `num_dims` dimensions of `x`."""
    return x.view(x.shape + (1,) * num_dims)


class VDM(nn.Module):
    def __init__(self, model, cfg, image_shape):
        super().__init__()
        self.model = model  # This is the UNetVDM
        self.cfg = cfg
        self.image_shape = image_shape
        self.vocab_size = 256
        
        if cfg.noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        elif cfg.noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {cfg.noise_schedule}")

    @property
    def device(self):
        return next(self.model.parameters()).device

    # --- SAMPLING FUNCTION ---
    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples):
        """Samples from p(z_s | z_t, x). Used for standard ancestral sampling."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        pred_noise = self.model(z, gamma_t)
        if clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t
            x_start.clamp_(-1.0, 1.0)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise)
        scale = sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z)

    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps, clip_samples):
        """The main sampling entry point."""
        z = torch.randn((batch_size, *self.image_shape), device=self.device)
        steps = linspace(1.0, 0.0, n_sample_steps + 1, device=self.device)
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples)
        logprobs = self.log_probs_x_z0(z_0=z)  # (B, C, H, W, vocab_size)
        x = argmax(logprobs, dim=-1)  # (B, C, H, W)
        return x.float() / (self.vocab_size - 1)  # normalize to [0, 1]
    # --- END OF SAMPLING ---

    def sample_q_t_0(self, x, times, noise=None):
        """Samples from the distributions q(x_t | x_0) at the given time steps."""
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t

    def sample_times(self, batch_size):
        if self.cfg.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times.requires_grad_(True) # Ensure times requires grad for loss

    def forward(self, batch, *, noise=None):
        """
        Refactored forward pass.
        Returns noise prediction and data dict for the loss function.
        """
        x, label = maybe_unpack_batch(batch)
        assert x.shape[1:] == self.image_shape
        assert 0.0 <= x.min() and x.max() <= 1.0
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))

        # Convert image to integers in range [0, vocab_size - 1].
        img_int = torch.round(x * (self.vocab_size - 1)).long()
        assert (img_int >= 0).all() and (img_int <= self.vocab_size - 1).all()
        # Check that the image was discrete with vocab_size values.
        assert allclose(img_int / (self.vocab_size - 1), x)

        # Rescale integer image to [-1 + 1/vocab_size, 1 - 1/vocab_size]
        x_clean = 2 * ((img_int + 0.5) / self.vocab_size) - 1

        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(x_clean.shape[0])
        if noise is None:
            noise = torch.randn_like(x_clean)
        x_t, gamma_t = self.sample_q_t_0(x=x_clean, times=times, noise=noise)

        # Forward through U-Net model
        model_out = self.model(x_t, gamma_t) # This is the noise prediction

        # Pack all other necessary tensors into a dict for the loss function
        batch_data = {
            "x_clean": x_clean,
            "noise": noise,
            "gamma_t": gamma_t,
            "times": times,
            "img_int": img_int,
            "bpd_factor": bpd_factor
        }

        return model_out, batch_data

    def log_probs_x_z0(self, x=None, z_0=None):
        """
        Computes log p(x | z_0) for all possible values of x.
        This is a helper function, now called by the VLB loss module.
        """
        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        if x is None and z_0 is not None:
            z_0_rescaled = z_0 / sqrt(sigmoid(-gamma_0))  # z_0 / alpha_0
        elif z_0 is None and x is not None:
            # Equal to z_0/alpha_0 with z_0 sampled from q(z_0 | x)
            z_0_rescaled = x + exp(0.5 * gamma_0) * torch.randn_like(x)  # (B, C, H, W)
        else:
            raise ValueError("Must provide either x or z_0, not both.")
        z_0_rescaled = z_0_rescaled.unsqueeze(-1)  # (B, C, H, W, 1)
        x_lim = 1 - 1 / self.vocab_size
        x_values = linspace(-x_lim, x_lim, self.vocab_size, device=self.device)
        logits = -0.5 * exp(-gamma_0) * (z_0_rescaled - x_values) ** 2  # broadcast x
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, C, H, W, vocab_size)
        return log_probs

# --- Noise Schedule Classes ---
class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        return self.b + self.w.abs() * t