"""
Stage 2 (Part A) - Automated Preprocessing Script for ProntoQA.

This script loads the trained VQ-VAE model (from Stage 1) and 
uses it to create the "assorted" dataset for training the 
main LLM (Stage 2). 

It specifically replaces logical reasoning hops with discrete latent 
tokens from the VQ-VAE codebook.
"""

import torch
import os
import sys
import json
from tqdm import tqdm

# --- Add 'src' to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
# -------------------------------------

from utils import (
    get_llm_tokenizer, 
    VQ_CODEBOOK_SIZE, 
    PATH_VQVAE_MODEL, 
    PATH_PROCESSED_DATA,
    create_assorted_dataset_prontoqa  # Updated for ProntoQA
)
from model.vae import VQVAEModel

# Define paths for raw ProntoQA data
PATH_RAW_TRAIN = os.path.join(project_root, 'data', 'prontoqa_5hop_train.json')

def main():
    print("--- 🚀 Starting Automated ProntoQA Assortment ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer = get_llm_tokenizer()
    vocab_size = len(tokenizer)

    # 2. Load trained VQ-VAE model (Stage 1)
    print(f"Loading VQ-VAE model from {PATH_VQVAE_MODEL}...")
    try:
        # Configuration must match the Stage 1 training parameters
        vq_model = VQVAEModel(
            vocab_size=vocab_size,
            d_model=256, 
            num_embeddings=VQ_CODEBOOK_SIZE,
            max_seq_len=128  # Optimized for short logical hops
        ).to(device)
        
        vq_model.load_state_dict(torch.load(PATH_VQVAE_MODEL, map_location=device))
        vq_model.eval()
        print("VQ-VAE model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: VQ-VAE weights not found at {PATH_VQVAE_MODEL}")
        return
    except Exception as e:
        print(f"Error loading VQ-VAE: {e}")
        return

    # 3. Load partitioned ProntoQA data
    print(f"Loading raw training data from {PATH_RAW_TRAIN}...")
    try:
        with open(PATH_RAW_TRAIN, 'r') as f:
            raw_data = json.load(f)
        # Ensure data is in list format
        data_list = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
    except FileNotFoundError:
        print(f"ERROR: Raw data not found at {PATH_RAW_TRAIN}")
        print("Please run 'python scripts/download_and_partition_prontoqa.py' first.")
        return

    # 4. Create the assorted dataset
    # This maps 1 logical hop -> 1 latent token
    assorted_samples = create_assorted_dataset_prontoqa(
        vq_model=vq_model,
        llm_tokenizer=tokenizer,
        data_list=data_list,
        device=device,
        max_hops_to_replace=3  # Randomized abstraction depth
    )
    
    if not assorted_samples:
        print("ERROR: No samples generated. Verify ProntoQA parsing logic.")
        return

    print(f"Generated {len(assorted_samples)} assorted logical chains.")

    # 5. Save the processed data for Stage 2 training
    os.makedirs(os.path.dirname(PATH_PROCESSED_DATA), exist_ok=True)
    
    print(f"Saving to {PATH_PROCESSED_DATA}...")
    with open(PATH_PROCESSED_DATA, 'w') as f:
        for item in tqdm(assorted_samples, desc="Writing JSONL"):
            f.write(json.dumps(item) + '\n')

    print("--- ✅ Preprocessing Complete ---")

if __name__ == "__main__":
    main()
