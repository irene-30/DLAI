"""
Defines the VQ-VAE model (Stage 1) for learning discrete text representations.
This is a Transformer Encoder/Decoder with a VQ bottleneck.
"""

import torch
import torch.nn as nn
from src.model.modules import VectorQuantizer # Import from modules.py

class VQVAEModel(nn.Module):
    """
    Transformer-based VQ-VAE Autoencoder.
    Implements the f_enc and f_dec described in the paper.
    
    Args:
        vocab_size (int): Size of the text tokenizer vocabulary.
        d_model (int): Internal dimension of the model.
        n_head (int): Number of attention heads.
        num_encoder_layers (int): Number of Transformer encoder layers.
        num_decoder_layers (int): Number of Transformer decoder layers.
        dim_feedforward (int): Hidden dim of feedforward networks.
        num_embeddings (int): Codebook size (K).
        commitment_cost (float): Beta for VQ loss.
        max_seq_len (int): Max sequence length for positional embeddings.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 n_head: int = 4, 
                 num_encoder_layers: int = 2, 
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 num_embeddings: int = 1024, 
                 commitment_cost: float = 1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.d_model = d_model

        # f_enc (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # VQ Bottleneck
        self.quantizer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
        
        # f_dec (Transformer Decoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Output layer to map back to vocabulary
        self.to_logits = nn.Linear(d_model, vocab_size)

    def encode(self, src_tokens: torch.Tensor):
        """Encodes text tokens into quantized vectors and indices."""
        # src_tokens shape: (B, T)
        B, T = src_tokens.shape
        
        # (B, T) -> (B, T, D)
        x = self.token_emb(src_tokens) * (self.d_model**0.5)
        x = x + self.pos_emb[:, :T, :]
        
        # (B, T, D) -> (B, T, D)
        encoded = self.transformer_encoder(x)
        
        # (B, T, D) -> (B, T, D), loss, (B, T)
        quantized, vq_loss, indices = self.quantizer(encoded)
        return quantized, vq_loss, indices

    def decode(self, quantized_memory: torch.Tensor, tgt_tokens: torch.Tensor):
        """Decodes quantized vectors back into text logits."""
        # tgt_tokens shape: (B, T)
        B, T = tgt_tokens.shape

        # Create decoder input embeddings
        tgt_emb = self.token_emb(tgt_tokens) * (self.d_model**0.5)
        tgt_emb = tgt_emb + self.pos_emb[:, :T, :]
        
        # Create a causal mask for the decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_emb.device)

        # Decode
        decoded = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=quantized_memory, 
            tgt_mask=tgt_mask
        )
        
        # (B, T, D) -> (B, T, VocabSize)
        logits = self.to_logits(decoded)
        return logits

    def forward(self, src_tokens: torch.Tensor):
        """Full autoencoding pass."""
        # src_tokens shape: (B, T)
        
        # Encode
        quantized, vq_loss, _ = self.encode(src_tokens)
        
        # Decode
        # We use src_tokens as the "target" for the decoder in an autoencoder setup
        logits = self.decode(quantized_memory=quantized, tgt_tokens=src_tokens)
        
        # Calculate Reconstruction Loss (Cross-Entropy)
        # We want the decoder to predict the *next* token, so we shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = src_tokens[:, 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss()
        recon_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Total loss = Reconstruction Loss + VQ Loss
        total_loss = recon_loss + vq_loss
        
        return total_loss, recon_loss, vq_loss
