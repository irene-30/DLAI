"""
Step 2 Module: RBF Quantizer for Post-Hoc Discretization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostHocRBFQuantizer(nn.Module):
    """
    This module learns a codebook to cluster a fixed/frozen continuous latent space.
    It does NOT use gradients from the reconstruction loss.
    It updates via EMA (Exponential Moving Average), which is equivalent to 
    Soft K-Means clustering.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, gamma: float = 10.0, decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.decay = decay
        
        # The Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        
        # EMA Buffers (internal state for K-Means)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def get_indices(self, z: torch.Tensor):
        """
        Returns the nearest codebook indices for a given continuous z.
        similarity = exp(-gamma * ||z - e||^2)
        """
        z_flat = z.reshape(-1, self.embedding_dim)
        
        # Squared L2 distance
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_flat, self.embedding.weight.t()))
        
        # RBF Similarity
        similarity = torch.exp(-self.gamma * distances)
        
        # Argmax to get discrete token
        encoding_indices = torch.argmax(similarity, dim=1)
        return encoding_indices

    def update(self, z: torch.Tensor):
        """
        Performs one step of Online K-Means / EMA update.
        This moves the codebook vectors towards the data 'z'.
        """
        z_flat = z.reshape(-1, self.embedding_dim)
        
        # 1. Assign points to current clusters
        indices = self.get_indices(z)
        
        # 2. One-hot encode assignments
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # 3. Calculate statistics for the batch
        n_assigned = encodings.sum(0) # How many points per cluster?
        sum_assigned = torch.matmul(encodings.t(), z_flat) # Sum of vectors per cluster
        
        # 4. Update buffers (EMA)
        self.cluster_size.data.mul_(self.decay).add_(n_assigned, alpha=1 - self.decay)
        self.ema_w.data.mul_(self.decay).add_(sum_assigned, alpha=1 - self.decay)
        
        # 5. Re-calculate centroids
        n = self.cluster_size.sum()
        # Laplace smoothing to avoid division by zero
        cluster_size_stabilized = (
            (self.cluster_size + 1e-5) / (n + self.num_embeddings + 1e-5) * n
        )
        
        weight_normalized = self.ema_w / cluster_size_stabilized.unsqueeze(1)
        
        # 6. Update the actual weights
        self.embedding.weight.data.copy_(weight_normalized)
        
        return indices
