"""
Step 1 Model: A Pure Continuous VAE.
"""
import torch
import torch.nn as nn

class ContinuousVAE(nn.Module):
    """
    Standard Transformer VAE with a continuous Gaussian latent space.
    
    It learns p(x|z) and q(z|x) where z is continuous.
    No quantization happens here.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 n_head: int = 4, 
                 num_encoder_layers: int = 2, 
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Projection to Gaussian Parameters (Mean and Log-Variance)
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.to_logits = nn.Linear(d_model, vocab_size)

    def reparameterize(self, mu, logvar):
        """
        z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, we can just use the mean
            return mu

    def encode(self, src_tokens):
        """
        Maps text tokens to continuous latent Z.
        """
        B, T = src_tokens.shape
        x = self.token_emb(src_tokens) * (self.d_model**0.5)
        x = x + self.pos_emb[:, :T, :]
        
        hidden = self.transformer_encoder(x)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

    def forward(self, src_tokens):
        # 1. Encode to continuous z
        z, mu, logvar = self.encode(src_tokens)
        
        # 2. Decode (reconstruct from continuous z)
        B, T = src_tokens.shape
        tgt_emb = self.token_emb(src_tokens) * (self.d_model**0.5)
        tgt_emb = tgt_emb + self.pos_emb[:, :T, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(z.device)
        
        decoded = self.transformer_decoder(tgt=tgt_emb, memory=z, tgt_mask=tgt_mask)
        logits = self.to_logits(decoded)
        
        # 3. Calculate Losses
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = src_tokens[:, 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss()
        recon_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (B * T)
        
        # Beta-VAE: We might need a small beta to keep reconstruction quality high
        kl_weight = 0.0001 
        total_loss = recon_loss + (kl_weight * kl_loss)
        
        return total_loss, recon_loss, kl_loss
