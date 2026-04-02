"""
Stage 2 Model: A Pure Continuous VAE.
Provides the smooth manifold required for Riemannian post-hoc discretization.
"""
import torch
import torch.nn as nn

class ContinuousVAE(nn.Module):
    """
    Standard Transformer VAE with a continuous Gaussian latent space.
    Used to implement the 'Latent-Oddity' (1710.11379) geometry-aware discretization.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 latent_dim: int = 128,
                 n_head: int = 4, 
                 num_encoder_layers: int = 2, 
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 128): # ProntoQA hops are typically < 128 tokens
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer Encoder (f_enc)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Continuous Bottleneck
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        # Mapping Z back to Decoder Dimension
        self.z_to_dec = nn.Linear(latent_dim, d_model)
        
        # Transformer Decoder (f_dec)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.to_logits = nn.Linear(d_model, vocab_size)

    def reparameterize(self, mu, logvar):
        """Standard reparameterization trick: z = mu + sigma * epsilon."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, src_tokens):
        """Maps text tokens to continuous latent Z (mu and logvar)."""
        B, T = src_tokens.shape
        x = self.token_emb(src_tokens) * (self.d_model**0.5)
        x = x + self.pos_emb[:, :T, :]
        
        hidden = self.transformer_encoder(x)
        
        # We take the mean of the sequence hidden states or the first token 
        # to represent the whole hop in the latent space.
        # Here we do it per-token to maintain sequence structure.
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

    def decode(self, z):
        """
        Pure decoding for Riemannian Metric calculation.
        Maps latent Z directly to logits without requiring ground-truth tokens.
        """
        # z shape: [Batch, T, Latent_Dim]
        z_projected = self.z_to_dec(z)
        
        # For the metric calculation (J^T * J), we can decode 
        # using z as the memory for a dummy sequence.
        # Here we return the linear projection for simplicity in metric estimation.
        return self.to_logits(z_projected)

    def forward(self, src_tokens):
        """Full autoencoding pass for training."""
        B, T = src_tokens.shape
        
        # 1. Encode
        z, mu, logvar = self.encode(src_tokens)
        
        # 2. Prepare Decoder Input
        tgt_emb = self.token_emb(src_tokens) * (self.d_model**0.5)
        tgt_emb = tgt_emb + self.pos_emb[:, :T, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(z.device)
        
        # Memory is the projected latent z
        z_mem = self.z_to_dec(z)
        
        # 3. Decode
        decoded = self.transformer_decoder(tgt=tgt_emb, memory=z_mem, tgt_mask=tgt_mask)
        logits = self.to_logits(decoded)
        
        # 4. Losses
        # Reconstruction: Next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = src_tokens[:, 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss()
        recon_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (B * T)
        
        # KL-Weight (Beta): Low value (0.0001) to prioritize reconstruction accuracy
        kl_weight = 0.0001 
        total_loss = recon_loss + (kl_weight * kl_loss)
        
        return total_loss, recon_loss, kl_loss
