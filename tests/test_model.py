import pytest
import torch
import sys
import os

# --- Add 'src' to the Python path ---
# This allows us to import from 'src.model', 'src.utils', etc.
# We go up one level (from 'tests' to the root) and then add 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
# -------------------------------------

# Now we can import our modules
from model.modules import VectorQuantizer
from model.vae import VQVAEModel
from model.transformer import get_llm_model
from utils import get_llm_tokenizer, LLM_MODEL_NAME

# --- Fixtures ---
# Fixtures create reusable components for our tests.

@pytest.fixture(scope="module")
def tokenizer():
    """A module-level fixture to get the tokenizer just once."""
    return get_llm_tokenizer()

@pytest.fixture(scope="module")
def test_config():
    """Provides a small, consistent config for all model tests."""
    return {
        "batch_size": 2,
        "seq_len": 32,
        "d_model": 16,
        "vocab_size": 50,
        "num_embeddings": 10,
        "n_head": 2
    }

# --- Tests ---

def test_vector_quantizer_forward(test_config):
    """
    Tests the VectorQuantizer module for correct I/O shapes.
    """
    B, T, D = test_config["batch_size"], test_config["seq_len"], test_config["d_model"]
    K = test_config["num_embeddings"]
    
    # 1. Arrange
    quantizer = VectorQuantizer(num_embeddings=K, embedding_dim=D, commitment_cost=0.25)
    dummy_input = torch.randn(B, T, D)
    
    # 2. Act
    quantized_ste, loss, indices = quantizer(dummy_input)
    
    # 3. Assert
    assert quantized_ste.shape == dummy_input.shape, "Quantized output shape is incorrect"
    assert loss.dim() == 0, "Loss should be a scalar"
    
    # Note: A good quantizer should return indices shaped (B, T)
    # Let's refine the test to check for this explicitly.
    # If your `src/model/modules.py` returns (B*T,), this test will fail
    # and you should update the module to return indices.view(B, T)
    assert indices.shape == (B, T), "Indices shape should be (Batch, SeqLen)"


def test_vqvae_model_forward(test_config, tokenizer):
    """
    Tests the full VQVAEModel (Stage 1) forward pass.
    """
    B, T = test_config["batch_size"], test_config["seq_len"]
    D = test_config["d_model"]
    K = test_config["num_embeddings"]
    # Use the *actual* tokenizer vocab size
    V = len(tokenizer) 
    
    # 1. Arrange
    model = VQVAEModel(
        vocab_size=V, 
        d_model=D, 
        n_head=test_config["n_head"], 
        num_embeddings=K,
        max_seq_len=T
    )
    # Create dummy token IDs (must be within vocab range)
    dummy_tokens = torch.randint(0, V, (B, T), dtype=torch.long)
    
    # 2. Act
    total_loss, recon_loss, vq_loss = model(dummy_tokens)
    
    # 3. Assert
    assert total_loss.dim() == 0, "Total loss should be a scalar"
    assert recon_loss.dim() == 0, "Recon loss should be a scalar"
    assert vq_loss.dim() == 0, "VQ loss should be a scalar"
    # Check that the loss is composed correctly
    assert torch.allclose(total_loss, recon_loss + vq_loss), "Total loss is not recon + vq"

def test_vqvae_model_encode_decode(test_config, tokenizer):
    """
    Tests the encode and decode methods of the VQVAEModel separately.
    """
    B, T = test_config["batch_size"], test_config["seq_len"]
    D = test_config["d_model"]
    K = test_config["num_embeddings"]
    V = len(tokenizer)
    
    # 1. Arrange
    model = VQVAEModel(
        vocab_size=V, 
        d_model=D, 
        n_head=test_config["n_head"], 
        num_embeddings=K,
        max_seq_len=T
    )
    dummy_tokens = torch.randint(0, V, (B, T), dtype=torch.long)
    
    # 2. Act
    quantized, vq_loss, indices = model.encode(dummy_tokens)
    logits = model.decode(quantized_memory=quantized, tgt_tokens=dummy_tokens)
    
    # 3. Assert
    assert quantized.shape == (B, T, D), "Quantized encoder output shape is wrong"
    assert vq_loss.dim() == 0, "VQ loss from encode should be scalar"
    assert indices.shape == (B, T), "Indices from encode should be (B, T)"
    assert logits.shape == (B, T, V), "Decoder output logits shape is wrong"


def test_llm_model_loading_and_resize(tokenizer):
    """
    Tests the `get_llm_model` helper to ensure it loads the model
    and correctly resizes the token embeddings.
    """
    # 1. Arrange
    new_vocab_size = len(tokenizer)
    
    # 2. Act
    model = get_llm_model(LLM_MODEL_NAME, tokenizer_len=new_vocab_size)
    
    # 3. Assert
    # Check both input and output embeddings to be sure
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    
    assert model.config.vocab_size == new_vocab_size, "Model config vocab size is not updated"
    assert input_embeddings.weight.shape[0] == new_vocab_size, "Input embedding size is not resized"
    assert output_embeddings.weight.shape[0] == new_vocab_size, "Output embedding size is not resized"