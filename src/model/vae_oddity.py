import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFVarianceNetwork(nn.Module):
    """
    Implementazione dell'Eq. 11 del paper Latent Space Oddity.
    Modella la precisione beta(z) in modo che decada a 0 (e la varianza tenda a infinito) 
    quando z si allontana dai centri dei dati, forzando la geodetica a stare vicino ai dati.
    """
    def __init__(self, latent_dim, num_centers=64, out_dim=256):
        super().__init__()
        self.num_centers = num_centers
        
        # Centri (c_k) e larghezza di banda (lambda_k) dei kernel RBF
        self.centers = nn.Parameter(torch.randn(num_centers, latent_dim))
        # Usiamo il logaritmo per garantire che lambda rimanga positivo durante il training
        self.log_lambdas = nn.Parameter(torch.zeros(num_centers)) 
        
        # Pesi W positivi (garantiti applicando abs() nel forward)
        self.W = nn.Parameter(torch.abs(torch.randn(out_dim, num_centers)))
        self.zeta = 1e-5 # Costante per prevenire la divisione per zero

    def forward(self, z):
        B, T, D = z.shape
        z_flat = z.reshape(-1, D) # [B*T, D]
        
        # Calcolo distanza al quadrato: ||z - c_k||^2
        dist_sq = torch.cdist(z_flat, self.centers)**2 # [B*T, K]
        
        # v_k(z) = exp(-\lambda_k * ||z - c_k||^2)
        lambdas = torch.exp(self.log_lambdas) # [K]
        v = torch.exp(-lambdas * dist_sq) # [B*T, K]
        
        # beta(z) = W * v(z) + zeta
        W_pos = torch.abs(self.W) # Forza pesi strettamente positivi
        beta = F.linear(v, W_pos) + self.zeta # [B*T, out_dim]
        
        # sigma^2 = 1 / beta => logvar = -log(beta)
        logvar_dec = -torch.log(beta)
        
        return logvar_dec.view(B, T, -1)


class ContinuousVAE(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 n_head: int = 4, 
                 num_encoder_layers: int = 2, 
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 256,
                 num_rbf_centers: int = 64):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        
        # Decoder (Generatore Stocastico)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Le due teste del decoder: Media (logits) e Varianza (RBF network)
        self.to_logits = nn.Linear(d_model, vocab_size)
        self.variance_head = RBFVarianceNetwork(latent_dim=d_model, num_centers=num_rbf_centers, out_dim=vocab_size)

    def encode(self, src_tokens):
        # Safety: if a dict is passed, extract the tensor
        if isinstance(src_tokens, dict):
            src_tokens = src_tokens['input_ids']
        B, T = src_tokens.shape
        x = self.token_emb(src_tokens) * (self.d_model**0.5)
        x = x + self.pos_emb[:, :T, :]
        hidden = self.transformer_encoder(x)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_from_z(self, z, tgt_tokens=None):
        """
        CRITICAL FOR LATENT ODDITY SCENARIO B: 
        Returns both the mean (logits) and variance (logvar_dec) for VJP calculations.
        """
        B, T, D = z.shape
        
        if tgt_tokens is None:
            # We use a zero-tensor for tgt_emb during Jacobian calculation 
            # to purely observe the functional mapping z -> output.
            tgt_emb = torch.zeros(B, T, self.d_model, device=z.device)
        else:
            tgt_emb = self.token_emb(tgt_tokens) * (self.d_model**0.5)
            tgt_emb = tgt_emb + self.pos_emb[:, :tgt_tokens.size(1), :]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(z.device)
        
        decoded = self.transformer_decoder(tgt=tgt_emb, memory=z, tgt_mask=tgt_mask)
        
        # Prevede Media (Logits)
        logits = self.to_logits(decoded)
        # Prevede Incertezza (Varianza via RBF)
        logvar_dec = self.variance_head(z)
        
        return logits, logvar_dec

    def forward(self, src_tokens):
        mu, logvar = self.encode(src_tokens)
        z = self.reparameterize(mu, logvar)
        
        logits, logvar_dec = self.decode_from_z(z, tgt_tokens=src_tokens)
        
        # Assicurati di aggiornare la loss function nel tuo training loop 
        # affinché ottimizzi il NLL pesato su `logvar_dec`!
        return logits, logvar_dec, mu, logvar
