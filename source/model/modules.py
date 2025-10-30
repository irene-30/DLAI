"""
Contains core, reusable neural network modules.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    """
    The core Vector Quantization (VQ) layer.
    
    This layer takes continuous encoder outputs and maps them to the
    nearest discrete vector in a learned "codebook".
    
    Args:
        num_embeddings (int): The number of vectors in the codebook (K).
        embedding_dim (int): The dimensionality of each vector (D).
        commitment_cost (float): The 'beta' hyperparameter for the commitment loss.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize the codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # inputs shape: (B, T, D)
        # B = Batch size, T = Sequence length, D = embedding_dim
        
        # Flatten input to (B*T, D)
        inputs_flat = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        # (B*T, D) -> (B*T, K)
        distances = (torch.sum(inputs_flat**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs_flat, self.embedding.weight.t()))
            
        # Find the nearest embedding indices
        # (B*T,)
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize: map indices back to embeddings
        # (B*T, D)
        quantized = self.embedding(encoding_indices)
        
        # Reshape back to (B, T, D)
        quantized = quantized.view_as(inputs)
        
        # --- Calculate VQ-VAE Losses (as in Eq. 2 from the paper) ---
        
        # 1. Codebook Loss (moves embeddings towards encoder outputs)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        # 2. Commitment Loss (moves encoder outputs towards embeddings)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-Through Estimator (STE)
        # For the backward pass, pass gradients from `quantized` to `inputs`
        quantized_ste = inputs + (quantized - inputs).detach()
        
        return quantized_ste, loss, encoding_indices