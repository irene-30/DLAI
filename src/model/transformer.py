"""
Defines the main LLM (Stage 2) and helper functions for loading it.
Includes weight initialization for new latent tokens.
"""

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

def get_llm_model(
    model_name: str, 
    tokenizer_len: int
) -> PreTrainedModel:
    """
    Loads a pretrained Causal LM (e.g., GPT-2) and resizes its token embeddings
    to accommodate the new latent tokens ([boLatent], [eoLatent], <latent_i>).
    
    Args:
        model_name (str): The name of the Hugging Face model.
        tokenizer_len (int): The new length of the tokenizer.
        
    Returns:
        PreTrainedModel: The loaded model with resized embeddings.
    """
    print(f"--- 🧩 Loading Stage 2 Reasoner: {model_name} ---")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 1. Check if resizing is necessary
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    
    if current_vocab_size != tokenizer_len:
        print(f"Resizing model token embeddings: {current_vocab_size} -> {tokenizer_len}")
        
        # 2. Perform the resize
        # This adds new rows to the embedding matrix (initialized randomly)
        model.resize_token_embeddings(tokenizer_len)
        
        # 3. Optional: Smart Initialization
        # Since latent tokens represent 'abstract reasoning', initializing them 
        # with the mean of the existing embeddings can provide a more stable 
        # starting point than pure random noise.
        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight
            # Calculate the mean of the original vocabulary
            mu = embeddings[:current_vocab_size].mean(dim=0)
            # Apply the mean to the newly added tokens
            embeddings[current_vocab_size:] = mu + 0.01 * torch.randn_like(embeddings[current_vocab_size:])
            
        print(f"Successfully initialized {tokenizer_len - current_vocab_size} new latent tokens.")
    
    # 4. Tie weights
    # For models like GPT-2, tying the word embeddings and the output weights 
    # ensures the model uses the same logic for 'understanding' and 'generating' latents.
    model.tie_weights()
    
    return model
