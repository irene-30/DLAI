"""
Contains core neural network modules for both Stage 1 (VQ-VAE) 
and Stage 2 (Continuous VAE with Riemannian Discretization).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    """
    Stage 1: The discrete Vector Quantization layer.
    Used for replicating the 'Token Assorted' baseline.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialization as per original VQ-VAE paper
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # inputs: (B, T, D)
        inputs_flat = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate L2 distances between inputs and codebook vectors
        distances = (torch.sum(inputs_flat**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs_flat, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view_as(inputs)
        
        # VQ Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-Through Estimator (STE)
        quantized_ste = inputs + (quantized - inputs).detach()
        
        return quantized_ste, loss, encoding_indices


class ReparameterizedGaussian(nn.Module):
    """
    Stage 2: The Continuous VAE Bottleneck.
    Implements the reparameterization trick to allow backpropagation 
    through a stochastic latent space.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # x is the output of the encoder (B, T, D) or (B, D)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        return z, kl_loss.mean(), mu, logvar
