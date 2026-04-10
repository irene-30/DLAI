import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_stochastic_metric_optimized(vae_model, src_tokens, n_samples=5):
    """
    Scenario B Implementation: Computes the full Metric Tensor G(z) = G_mu + G_sigma
    using Vector-Jacobian Products (VJP) for optimal memory performance on Colab.
    """
    if isinstance(src_tokens, dict):
       src_tokens = src_tokens['input_ids']

    # 1. Re-run encoder to get mu with a fresh graph
    mu, _ = vae_model.encode(src_tokens)
    
    # 2. Define z as a differentiable leaf node based on mu
    z = mu.detach().requires_grad_(True)
    
    B, T, D = z.shape
    T_sub = min(T, 16)
    
    # 3. Get both mean and variance from the decoder
    logits, logvar_dec = vae_model.decode_from_z(z[:, :T_sub, :])
    
    G_mu = torch.zeros(B, T_sub, D, D, device=z.device)
    G_sigma = torch.zeros(B, T_sub, D, D, device=z.device)

    # --- Term 1: G_mu (Distortion of the Mean) ---
    for _ in range(n_samples):
        v_mu = torch.randn_like(logits)
        vjp_mu = torch.autograd.grad(logits, z, grad_outputs=v_mu, retain_graph=True)[0]
        vjp_mu = vjp_mu[:, :T_sub, :]
        G_mu += torch.einsum('btd, btk -> btdk', vjp_mu, vjp_mu)
    
    G_mu /= n_samples

    # --- Term 2: G_sigma (Distortion of the Uncertainty) ---
    # Convert logvar to standard deviation (sigma)
    sigma_dec = torch.exp(0.5 * logvar_dec)
    
    for _ in range(n_samples):
        v_sig = torch.randn_like(sigma_dec)
        
        # Now this works because sigma_dec is derived directly from z via the RBF network
        vjp_sig = torch.autograd.grad(sigma_dec, z, grad_outputs=v_sig, retain_graph=True)[0]
        vjp_sig = vjp_sig[:, :T_sub, :]
        
        G_sigma += torch.einsum('btd, btk -> btdk', vjp_sig, vjp_sig)
        
    G_sigma /= n_samples

    return (G_mu + G_sigma).detach()


class LatentOddityQuantizer(nn.Module):
    def __init__(self, vae_model, num_embeddings, embedding_dim, gamma=10.0, decay=0.99):
        super().__init__()
        self.vae_model = vae_model  # ContinuousVAE
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.decay = decay
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def get_indices(self, z, input_ids):
        """
        Calculates indices using the Riemannian distance metric G(z).
        Args:
            z (Tensor): The latent representation [B, T, D]
            input_ids (Tensor): Original tokens [B, T] needed to compute G(z).
        """
        # Step A: Compute the Full Stochastic Metric Tensor
        G = compute_stochastic_metric_optimized(self.vae_model, input_ids)
        
        # Step B: Compute Riemannian Distance
        B, T, D = z.shape
        T_sub = G.size(1) 
        z_sub = z[:, :T_sub, :]
        
        z_flat = z_sub.reshape(-1, D)          # (B*T_sub, D)
        G_flat = G.reshape(-1, D, D)           # (B*T_sub, D, D)
        embeddings = self.embedding.weight     # (K, D)

        # Difference: (B*T_sub, K, D)
        diff = z_flat.unsqueeze(1) - embeddings.unsqueeze(0)
        
        # Mahalanobis product: diff^T @ G @ diff
        G_diff = torch.matmul(diff, G_flat.unsqueeze(1))
        distances = torch.sum(G_diff * diff, dim=-1)

        # Step C: RBF Similarity
        similarity = torch.exp(-self.gamma * distances)
        return torch.argmax(similarity, dim=1)

    def update(self, z, input_ids):
        """
        Riemannian Update: Cluster using the decoder's functional geometry.
        Args:
            z (Tensor): The latent mean [B, T, D].
            input_ids (Tensor): Raw tokens needed to re-run the metric calculation.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided for the Latent Oddity update.")
            
        # 1. Compute the local metric tensor G(z) = G_mu + G_sigma
        G = compute_stochastic_metric_optimized(self.vae_model, input_ids)
    
        # 2. Prepare tensors
        T_sub = G.size(1) 
        z_sub = z[:, :T_sub, :]
    
        z_flat = z_sub.reshape(-1, self.embedding_dim)
        G_flat = G.reshape(-1, self.embedding_dim, self.embedding_dim)
    
        # 3. Compute Riemannian distance: d^2 = (z - e)^T G (z - e)
        diff = z_flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0)
    
        G_diff = torch.matmul(diff, G_flat.unsqueeze(1)) 
        dist = torch.sum(G_diff * diff, dim=-1) 
    
        # 4. Find nearest codebook entries
        indices = torch.argmin(dist, dim=1)
    
        # 5. Exponential Moving Average (EMA) Update
        encodings = torch.nn.functional.one_hot(indices, self.num_embeddings).float()
    
        self.cluster_size.data.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
    
        dw = torch.matmul(encodings.t(), z_flat)
        self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
    
        # Laplacian smoothing
        n = self.cluster_size.sum()
        smoothed_cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        self.embedding.weight.data.copy_(self.ema_w / smoothed_cluster_size.unsqueeze(1))
    
        return indices
