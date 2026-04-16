import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperbolicVAE(nn.Module):
    """
    Transformer VAE con spazio latente nella Palla di Poincaré.
    Sostituisce la logica Euclidea con proiezioni iperboliche per preservare 
    le gerarchie del ragionamento matematico.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 n_head: int = 4, 
                 num_encoder_layers: int = 2, 
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 512,
                 c: float = 1.0): # Curvatura dello spazio
        super().__init__()
        
        self.d_model = d_model
        self.c = c 
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Proiezioni verso lo spazio tangente iperbolico
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.to_logits = nn.Linear(d_model, vocab_size)

    def exp_map0(self, v):
        """Mappa esponenziale: proietta dallo spazio tangente 0 alla Palla di Poincaré."""
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        # Formula: tanh(sqrt(c) * |v|) * v / (sqrt(c) * |v|)
        return torch.tanh(self.c**0.5 * norm_v) * v / (self.c**0.5 * norm_v)

    def reparameterize(self, mu_tangent, logvar):
        """
        Campionamento iperbolico: campiona in Euclideo e proietta sulla varietà.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_tangent = mu_tangent + eps * std
            return self.exp_map0(z_tangent)
        else:
            return self.exp_map0(mu_tangent)

    def encode(self, src_tokens):
        B, T = src_tokens.shape
        x = self.token_emb(src_tokens) * (self.d_model**0.5)
        x = x + self.pos_emb[:, :T, :]
        
        hidden = self.transformer_encoder(x)
        
        mu_tangent = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z_hyp = self.reparameterize(mu_tangent, logvar)
        
        return z_hyp, mu_tangent, logvar

    def forward(self, src_tokens):
        # 1. Encode verso lo spazio latente iperbolico Z
        z_hyp, mu_tangent, logvar = self.encode(src_tokens)
        
        # 2. Decode: Il decoder riceve coordinate iperboliche (|z| < 1)
        B, T = src_tokens.shape
        tgt_emb = self.token_emb(src_tokens) * (self.d_model**0.5)
        tgt_emb = tgt_emb + self.pos_emb[:, :T, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(z_hyp.device)
        
        # Il Transformer Decoder elabora z_hyp come memory
        decoded = self.transformer_decoder(tgt=tgt_emb, memory=z_hyp, tgt_mask=tgt_mask)
        logits = self.to_logits(decoded)
        
        # 3. Calcolo delle Loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = src_tokens[:, 1:].contiguous()
        
        recon_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # KL Divergence Iperbolica (Approssimazione per Wrapped Normal)
        # Include il log-determinante per compensare la curvatura della mappa esponenziale
        r = mu_tangent.norm(dim=-1, keepdim=True)
        # Fattore di correzione geometrica: (sinh(sqrt(c)*r) / (sqrt(c)*r))^(d-1)
        # Qui semplificato come penalità sulla norma per spingere verso il centro (root)
        kl_geom_adj = (self.d_model - 1) * torch.log(torch.sinh(self.c**0.5 * r + 1e-7) / (self.c**0.5 * r + 1e-7))
        
        kl_eucl = -0.5 * torch.sum(1 + logvar - mu_tangent.pow(2) - logvar.exp(), dim=-1)
        kl_loss = (kl_eucl + kl_geom_adj.squeeze(-1)).mean()
        
        kl_weight = 0.0001 
        total_loss = recon_loss + (kl_weight * kl_loss)
        
        return total_loss, recon_loss, kl_loss
