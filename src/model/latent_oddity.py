import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_stochastic_metric_optimized(vae_model, z, n_samples=5):
    """
    Optimized for Colab: Computes the Metric Tensor G(z) using 
    Vector-Jacobian Products (VJP) to save memory.
    """
    # 1. Re-run encoder to get mu/logvar with a fresh graph
    # We do NOT use torch.no_grad() here
    mu, logvar = vae_model.encode(input_ids)
    
    # 2. Define z as a differentiable leaf node based on mu
    z = mu.detach().requires_grad_(True)
    
    B, T, D = z.shape
    T_sub = min(T, 16)
    
    # --- G_mu (Distortion) ---
    logits = vae_model.decode_from_z(z[:, :T_sub, :])
    G = torch.zeros(B, T_sub, D, D, device=z.device)

    
    # Sample a few output dimensions to estimate the metric
    # This is a 'Stochastic' approximation of the Stochastic Metric
    for _ in range(n_samples):
        # Random projection vector
        v = torch.randn_like(logits)
        
        # Vector-Jacobian Product
        # This is much faster and memory-efficient than a full Jacobian
        vjp = torch.autograd.grad(logits, z, grad_outputs=v, retain_graph=True)[0]
        vjp = vjp[:, :T_sub, :]
        
        # Outer product to contribute to the metric tensor
        G_mu += torch.einsum('btd, btk -> btdk', vjp, vjp)
    
    G_mu /= n_samples

    # 3. Compute Uncertainty G_sigma
    # grad_sigma: how the predicted variance changes with respect to z
    sigma = torch.exp(0.5 * logvar[:, :T_sub, :])
    #G_sigma = torch.zeros_like(G_mu)
    
    for d in range(D):
        # This will now work because logvar -> sigma -> grad_s is a valid chain
        grad_s = torch.autograd.grad(outputs = sigma[:, :, d].sum(), inputs = z, retain_graph=True)[0]
        grad_s = grad_s[:, :T_sub, :]
        G_sigma += torch.einsum('btd, btk -> btdk', grad_s, grad_s)

    return (G_mu + G_sigma).detach()

class LatentOddityQuantizer(nn.Module):
    def __init__(self, vae_model, num_embeddings, embedding_dim, gamma=10.0, decay=0.99):
        super().__init__()
        self.vae_model = vae_model  # Your ContinuousVAE
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.decay = decay
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def get_indices(self, z, logvar):
        """
        WHERE THE MAGIC HAPPENS:
        1. Compute G(z) using the optimized Jacobian function.
        2. Use G(z) to calculate Riemannian distances.
        """
        # Step A: Compute the Metric Tensor (The Stochastic 'Oddity' term)
        # We call the optimized function here
        G = compute_stochastic_metric_optimized(self.vae_model, z, logvar)
        
        # Step B: Compute Riemannian Distance
        # dist = (z - e)^T * G * (z - e)
        # Because z is (B, T, D) and G is (B, T, D, D), we process token-by-token
        B, T, D = z.shape
        z_flat = z.view(-1, D)          # (B*T, D)
        G_flat = G.view(-1, D, D)        # (B*T, D, D)
        embeddings = self.embedding.weight # (K, D)

        # Difference between every z token and every codebook entry
        # diff: (B*T, K, D)
        diff = z_flat.unsqueeze(1) - embeddings.unsqueeze(0)
        
        # Mahalanobis product: diff^T @ G @ diff
        # G_diff: (B*T, K, D)
        G_diff = torch.matmul(diff, G_flat.unsqueeze(1))
        # distances: (B*T, K)
        distances = torch.sum(G_diff * diff, dim=-1)

        # Step C: RBF Similarity
        similarity = torch.exp(-self.gamma * distances)
        return torch.argmax(similarity, dim=1)

    def update(self, z, logvar):
        """Standard EMA update using the new Riemannian indices."""
        indices = self.get_indices(z, logvar)
        
        # ... (rest of the EMA update logic from your previous class) ...
        return indices
