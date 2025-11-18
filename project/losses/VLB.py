import torch
from torch import nn, autograd, sigmoid
from torch.special import expm1

# Helper function from VDM for the Kullback-Leibler divergence, needed in the loss
def kl_std_normal(mean_squared, var):
    """
    KL divergence between N(mean_squared, var) and N(0, 1).
    """
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)

class VLB(nn.Module):
    """
    Computes the Variational Lower Bound (VLB) loss for the VDM.
    
    This loss module is initialized with the VDM model itself to gain
    access to shared components like the gamma schedule and helper functions.
    """
    def __init__(self, vdm_model):
        super().__init__()
        # Store references to components from the VDM
        self.vdm = vdm_model 

    def forward(self, model_out, batch_data):
        """
        Calculates the VLB loss.
        
        Args:
            model_out (torch.Tensor): The noise prediction from the U-Net.
            batch_data (dict): A dict containing all other tensors needed,
                               provided by the VDM.forward() method.
        """
        # Unpack all the required tensors from the batch_data dict
        x_clean = batch_data["x_clean"]
        noise = batch_data["noise"]
        gamma_t = batch_data["gamma_t"]
        times = batch_data["times"]
        img_int = batch_data["img_int"]
        bpd_factor = batch_data["bpd_factor"]
        
        # --- Start of VLB Loss ---

        # *** Diffusion loss (bpd)
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = ((model_out - noise) ** 2).sum((1, 2, 3))  # (B, )
        diffusion_loss = 0.5 * pred_loss * gamma_grad * bpd_factor

        # *** Latent loss (bpd): KL divergence from N(0, 1) to q(z_1 | x)
        gamma_1 = self.vdm.gamma(torch.tensor([1.0], device=self.vdm.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x_clean**2  # (alpha_1 * x)**2
        latent_loss = kl_std_normal(mean_sq, sigma_1_sq).sum((1, 2, 3)) * bpd_factor

        # *** Reconstruction loss (bpd): - E_{q(z_0 | x)} [log p(x | z_0)].
        # We use the helper function from the VDM model
        log_probs = self.vdm.log_probs_x_z0(x=x_clean)  # (B, C, H, W, vocab_size)
        
        # One-hot representation of original image. Shape: (B, C, H, W, vocab_size).
        x_one_hot = torch.zeros((*x_clean.shape, self.vdm.vocab_size), device=self.vdm.device)
        x_one_hot.scatter_(4, img_int.unsqueeze(-1), 1)  # one-hot over last dim
        
        # Select the correct log probabilities.
        log_probs = (x_one_hot * log_probs).sum(-1)  # (B, C, H, W)
        
        # Overall logprob for each image in batch.
        recons_loss = -log_probs.sum((1, 2, 3)) * bpd_factor

        # *** Overall loss in bpd. Shape (B, ).
        loss = diffusion_loss + latent_loss + recons_loss

        with torch.no_grad():
            gamma_0 = self.vdm.gamma(torch.tensor([0.0], device=self.vdm.device))
        
        metrics = {
            "bpd": loss.mean(),
            "diff_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "loss_recon": recons_loss.mean(),
            "gamma_0": gamma_0.item(),
            "gamma_1": gamma_1.item(),
        }
        
        return loss.mean(), metrics