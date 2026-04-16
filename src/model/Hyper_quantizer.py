import torch
import torch.nn as nn
import torch.nn.functional as F

# Cruciale per operazioni geometriche avanzate e ottimizzazione su varietà
# Se non disponibile, le funzioni di distanza devono essere implementate manualmente
try:
    import geoopt
except ImportError:
    geoopt = None

# Per il calcolo di centroidi di Fréchet e stabilità numerica
import math

class HyperbolicPostHocQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        
        # Inizializzazione vicina all'origine (gerarchia radice)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1e-3, 1e-3)
        
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def poincare_dist(self, u, v):
        """Calcola la distanza iperbolica tra u e v nella palla di Poincaré."""
        sq_dist = torch.sum((u - v)**2, dim=-1)
        r_u = 1 - torch.sum(u**2, dim=-1)
        r_v = 1 - torch.sum(v**2, dim=-1)
        arg = 1 + 2 * sq_dist / (r_u * r_v)
        return torch.acosh(arg.clamp(min=1 + 1e-7))

    def get_indices(self, z):
        # z: [B, T, D] -> [B*T, 1, D] | centroids: [1, K, D]
        z_flat = z.view(-1, self.embedding_dim).unsqueeze(1)
        centroids = self.embedding.weight.unsqueeze(0)
        
        # Distanza iperbolica invece di L2
        distances = self.poincare_dist(z_flat, centroids)
        return torch.argmin(distances, dim=1)

    def update(self, z):
        """Aggiornamento EMA iperbolico (Approssimazione del Fréchet Mean)."""
        z_flat = z.reshape(-1, self.embedding_dim)
        indices = self.get_indices(z)
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # Statistiche Batch
        n_assigned = encodings.sum(0)
        sum_assigned = torch.matmul(encodings.t(), z_flat)
        
        # Update EMA
        self.cluster_size.data.mul_(self.decay).add_(n_assigned, alpha=1 - self.decay)
        self.ema_w.data.mul_(self.decay).add_(sum_assigned, alpha=1 - self.decay)
        
        # Calcolo dei nuovi centroidi (Media di Fréchet approssimata)
        # In spazi iperbolici, la media non è una semplice divisione.
        # Una tecnica comune è normalizzare e riportare sulla palla.
        new_weights = self.ema_w / (self.cluster_size.unsqueeze(1) + 1e-5)
        # Clipping per garantire che rimangano dentro la palla di Poincaré (|z| < 1)
        norm = new_weights.norm(dim=-1, keepdim=True)
        max_norm = 0.999
        new_weights = new_weights * torch.min(norm, torch.ones_like(norm) * max_norm) / (norm + 1e-10)
        
        self.embedding.weight.data.copy_(new_weights)
        return indices
