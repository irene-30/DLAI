"""
Defines the main LLM (Stage 2) and helper functions for loading it.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import List

def get_llm_model(
    model_name: str, 
    tokenizer_len: int
) -> PreTrainedModel:
    """
    Loads a pretrained Causal LM and resizes its token embeddings
    to accommodate the new latent tokens.
    
    Args:
        model_name (str): The name of the Hugging Face model (e.g., "gpt2").
        tokenizer_len (int): The new length of the tokenizer (including latent tokens).
        
    Returns:
        PreTrainedModel: The loaded model with resized embeddings.
    """
    print(f"Loading base LLM: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # --- CRITICAL STEP ---
    # Resize the model's token embeddings to match the new tokenizer size.
    # This adds new, randomly initialized vectors for our latent/special tokens.
    current_vocab_size = model.config.vocab_size
    if current_vocab_size != tokenizer_len:
        print(f"Resizing model token embeddings from {current_vocab_size} to {tokenizer_len}")
        model.resize_token_embeddings(tokenizer_len)
    
    return model