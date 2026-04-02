"""
Step 2 Module: Riemannian Post-Hoc Quantizer.
Implements the 'Latent-Oddity' (1710.11379) metric for geometry-aware discretization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostHocRiemannianQuantizer(nn.Module):
    """
    This module performs discretization using a Stochastic Riemannian Metric.
    It weights the distance between a latent vector 'z' and codebook centroids 
    by the 'sensitivity' of the decoder at that point.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, n_stochastic_samples: int = 5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.n_samples = n_stochastic_samples
        
        # The Codebook (Centroids)
        # These are usually initialized via K-Means on the VAE's mu outputs
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()

    def _compute_riemannian_weight(self, vae_model: nn.Module, z: torch.Tensor):
        """
        Approximates the local metric G(z) using the Vector-Jacobian Product.
        High weight = high sensitivity (small Z change = large Text change).
        """
        z_ref = z.detach().clone().requires_grad_(True)
        
        # We need gradients to compute the Jacobian-based weight
        with torch.set_grad_enabled(True):
            # Decode z to logits
            logits = vae_model.decode(z_ref)
            
            weight = 0
            for _ in range(self.n_samples):
                # Stochastic projection (Hutchinson's trace estimator logic)
                v = torch.randn_like(logits)
                logits.backward(v, retain_graph=True)
                
                # The norm of the gradient wrt z reflects the local 'curvature'
                weight += z_ref.grad.norm(p=2, dim=-1)
                z_ref.grad.zero_()
                
        # Return a scalar weight per batch element
        return (weight / self.n_samples).detach()

    def get_indices(self, z: torch.Tensor, vae_model: nn.Module):
        """
        Finds the best discrete token using the Riemannian distance:
        d_R(z, e) = ||z - e|| * Weight_Riemannian(z)
        """
        # 1. Compute the local metric weight
        # This tells us how 'important' distance is at this specific location in Z
        g_weight = self._compute_riemannian_weight(vae_model, z) # Shape: [Batch]
        
        # 2. Calculate Euclidean distances to all codebook entries
        # z: [Batch, D], centroids: [K, D]
        z_flat = z.view(-1, self.embedding_dim)
        
        # (Batch, K)
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_flat, self.embedding.weight.t()))
        
        # 3. Apply the Riemannian correction
        # We multiply the Euclidean distance by the decoder's sensitivity
        riemannian_distances = distances * g_weight.unsqueeze(1)
        
        # 4. Return the functionally closest index
        return torch.argmin(riemannian_distances, dim=1)

    def initialize_codebook(self, z_samples: torch.Tensor):
        """
        Standard K-Means initialization to start centroids at data centers.
        z_samples: A large batch of encoder outputs [N, D]
        """
        from sklearn.cluster import KMeans
        print(f"Initializing {self.num_embeddings} centroids via K-Means...")
        kmeans = KMeans(n_clusters=self.num_embeddings, n_init=10)
        kmeans.fit(z_samples.cpu().numpy())
        self.embedding.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
