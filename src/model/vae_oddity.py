import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousVAE(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 n_head: int = 4, 
                 num_encoder_layers: int = 2, 
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 256):
        super().__init__()
        
        self.d_model = d_model
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
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.to_logits = nn.Linear(d_model, vocab_size)

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
        CRITICAL FOR LATENT ODDITY: 
        This method allows us to compute the Jacobian of the output with respect to Z.
        """
        B, T, D = z.shape
        
        # If no target tokens provided (e.g., during Jacobian calc), use a dummy/start sequence
        if tgt_tokens is None:
            # We use a zero-tensor or a fixed 'start' token embedding to see how 
            # variations in Z affect the transformer's internal state.
            tgt_emb = torch.zeros(B, T, self.d_model).to(z.device)
        else:
            tgt_emb = self.token_emb(tgt_tokens) * (self.d_model**0.5)
            tgt_emb = tgt_emb + self.pos_emb[:, :tgt_tokens.size(1), :]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(z.device)
        
        # Memory is Z. We observe how the decoder 'attends' to Z.
        decoded = self.transformer_decoder(tgt=tgt_emb, memory=z, tgt_mask=tgt_mask)
        logits = self.to_logits(decoded)
        return logits

    def forward(self, src_tokens):
        mu, logvar = self.encode(src_tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_from_z(z, tgt_tokens=src_tokens)
        
        # Loss calculation logic...
        return logits, mu, logvar
