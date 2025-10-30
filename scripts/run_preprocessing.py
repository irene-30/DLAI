"""
Stage 2 (Part A) - Preprocessing Script.

This script loads the trained VQ-VAE model (from Stage 1) and
uses it to create the "assorted" dataset for training the
main LLM (Stage 2).

It replaces the '03_preprocessing_assorted.ipynb' notebook
for automated pipelines.
"""

import torch
import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Add 'src' to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
# -------------------------------------

from utils import (
    get_llm_tokenizer, MAX_SEQ_LEN, VQ_CODEBOOK_SIZE, 
    PATH_VQVAE_MODEL, PATH_PROCESSED_DATA,
    create_assorted_dataset
)
from model.vae import VQVAEModel

# Define paths for raw data (relative to project root)
PATH_RAW_TRAIN = os.path.join(project_root, 'data', 'raw', 'gsm8k_train.jsonl')

def main():
    print("--- 🚀 Starting Assorted Dataset Preprocessing ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = get_llm_tokenizer()
    vocab_size = len(tokenizer)

    # 2. Load trained VQ-VAE model
    print(f"Loading VQ-VAE model from {PATH_VQVAE_MODEL}...")
    try:
        # Note: Parameters MUST match the trained model
        d_model = 256 # This should match the config in src/train.py or notebook 02
        
        vq_model = VQVAEModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_embeddings=VQ_CODEBOOK_SIZE,
            max_seq_len=MAX_SEQ_LEN
        ).to(device)
        
        vq_model.load_state_dict(torch.load(PATH_VQVAE_MODEL, map_location=device))
        vq_model.eval()
        print("VQ-VAE model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: VQ-VAE model not found at {PATH_VQVAE_MODEL}")
        print("Please run 'python src/train.py vqvae' first.")
        return
    except Exception as e:
        print(f"Error loading VQ-VAE model: {e}")
        return

    # 3. Load raw dataset
    print(f"Loading raw training data from {PATH_RAW_TRAIN}...")
    try:
        raw_dataset = load_dataset("json", data_files=PATH_RAW_TRAIN, split="train")
    except FileNotFoundError:
        print(f"ERROR: Raw data not found at {PATH_RAW_TRAIN}")
        print("Please run 'python scripts/download_data.py' first.")
        return

    # 4. Create the assorted dataset
    # This function is imported from src/utils.py
    assorted_samples = create_assorted_dataset(
        vq_model=vq_model,
        llm_tokenizer=tokenizer,
        dataset=raw_dataset
    )
    
    if not assorted_samples:
        print("ERROR: No samples were generated. Check raw data and parsing logic.")
        return

    print(f"\nGenerated {len(assorted_samples)} assorted samples.")

    # 5. Save the processed data
    output_dir = os.path.dirname(PATH_PROCESSED_DATA)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving processed data to {PATH_PROCESSED_DATA}...")
    with open(PATH_PROCESSED_DATA, 'w') as f:
        for item in tqdm(assorted_samples, desc="Writing to file"):
            f.write(json.dumps(item) + '\n')

    print("--- ✅ Preprocessing Complete ---")

if __name__ == "__main__":
    main()
