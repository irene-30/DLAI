"""
Defines the VQ-VAE model (Stage 1) for learning discrete text representations.
This model acts as the 'compressor' that maps text hops to latent tokens.
"""

import torch
import torch.nn as nn
from src.model.modules import VectorQuantizer

class VQVAEModel(nn.Module):
    """
    Transformer-based VQ-VAE Autoencoder for logical reasoning hops.
    
    The Encoder (f_enc) maps a text hop into a continuous latent space,
    which is then quantized by the codebook. The Decoder (f_dec) attempts 
    to reconstruct the original text hop from that quantized vector.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 n_head: int = 4, 
                 num_encoder_layers: int = 2, 
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 num_embeddings: int = 1024, 
                 commitment_cost: float = 0.25, # Standard beta from VQ-VAE papers
                 max_seq_len: int = 128): # ProntoQA hops are short
        super().__init__()
        
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # f_enc: Maps text tokens to a latent bottleneck
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # VQ Bottleneck: The discrete codebook
        self.quantizer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
        
        # f_dec: Reconstructs text from the quantized latent vector
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Final output head for reconstruction
        self.to_logits = nn.Linear(d_model, vocab_size)

    def encode(self, src_tokens: torch.Tensor):
        """Encodes reasoning hops into quantized vectors and discrete indices."""
        B, T = src_tokens.shape
        
        # Embedding + Positional Encoding
        x = self.token_emb(src_tokens) * (self.d_model**0.5)
        x = x + self.pos_emb[:, :T, :]
        
        # Transformer pass
        encoded = self.transformer_encoder(x)
        
        # Quantization bottleneck
        # quantized: (B, T, D), vq_loss: scalar, indices: (B, T)
        quantized, vq_loss, indices = self.quantizer(encoded)
        return quantized, vq_loss, indices

    def decode(self, quantized_memory: torch.Tensor, tgt_tokens: torch.Tensor):
        """Decodes the quantized bottleneck back into text logits."""
        B, T = tgt_tokens.shape

        tgt_emb = self.token_emb(tgt_tokens) * (self.d_model**0.5)
        tgt_emb = tgt_emb + self.pos_emb[:, :T, :]
        
        # Causal mask for the auto-regressive reconstruction
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_emb.device)

        # The 'memory' here is the quantized output of the encoder
        decoded = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=quantized_memory, 
            tgt_mask=tgt_mask
        )
        
        return self.to_logits(decoded)

    def forward(self, src_tokens: torch.Tensor):
        """
        Calculates the total training loss.
        loss = Reconstruction (Cross-Entropy) + Quantization (Commitment)
        """
        # 1. Encode to the discrete bottleneck
        quantized, vq_loss, _ = self.encode(src_tokens)
        
        # 2. Decode to reconstruct original text
        # We use the src_tokens as target for autoencoding
        logits = self.decode(quantized_memory=quantized, tgt_tokens=src_tokens)
        
        # 3. Calculate Reconstruction Loss (Cross-Entropy)
        # Shift tokens for next-word prediction logic
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = src_tokens[:, 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss()
        recon_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        total_loss = recon_loss + vq_loss
        
        return total_loss, recon_loss, vq_loss

    def get_codebook_weights(self):
        """Returns the learned codebook centroids."""
        return self.quantizer.embedding.weight.data
