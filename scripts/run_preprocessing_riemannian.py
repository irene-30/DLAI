"""
Stage 2 (Part B) - Riemannian Preprocessing Script.
Loads Continuous VAE -> Computes Stochastic Pull-back Metric -> Creates Assorted Dataset.

This script implements the geometry-aware discretization proposed in the 
'Latent-Oddity' (arXiv:1710.11379) research to protect logical boundaries.
"""
import torch
import os
import sys
import json
from tqdm import tqdm

# --- Add 'src' to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from utils import (
    get_llm_tokenizer, 
    MAX_SEQ_LEN, 
    VQ_CODEBOOK_SIZE, 
    PATH_PROCESSED_DATA,
    create_assorted_dataset_riemannian,
    PATH_RAW_DATA
)
from model.vae_continuous import ContinuousVAE

# Path to the trained continuous manifold from Stage 2 Part A
PATH_CONTINUOUS_VAE = os.path.join(project_root, "experiments/vae_continuous.pth")

def main():
    print("--- 📐 Starting Riemannian Assorted Dataset Preprocessing ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = get_llm_tokenizer()

    # 1. Load Frozen Continuous VAE
    # This model provides the differentiable manifold for Jacobian calculation
    print(f"Loading Continuous VAE from {PATH_CONTINUOUS_VAE}...")
    vae = ContinuousVAE(
        vocab_size=len(tokenizer), 
        d_model=256, 
        latent_dim=128, 
        max_seq_len=128
    ).to(device)
    
    try:
        vae.load_state_dict(torch.load(PATH_CONTINUOUS_VAE, map_location=device))
        vae.eval()
        print("VAE manifold loaded successfully.")
    except FileNotFoundError:
        print("Error: Continuous VAE not found. Please run 'train_continuous_vae.py' first.")
        return

    # 2. Setup Codebook Centroids
    # Note: For Riemannian discretization, centroids are often initialized via K-Means 
    # on the encoded 'mu' vectors of the training set
    print("Initializing geometry-aware centroids...")
    # Placeholder for centroids; in practice, load these from a pre-calculated .pth file
    centroids = torch.randn(VQ_CODEBOOK_SIZE, 128).to(device) 

    # 3. Load Raw ProntoQA Data
    # Local JSON containing the 5-hop reasoning chains
    print(f"Loading raw ProntoQA data from {PATH_RAW_DATA}...")
    try:
        with open(PATH_RAW_DATA, 'r') as f:
            raw_json = json.load(f)
        data_list = list(raw_json.values()) if isinstance(raw_json, dict) else raw_json
    except FileNotFoundError:
        print("Error: Raw ProntoQA JSON not found.")
        return

    # 4. Create Riemannian Assorted Data
    # This uses the Stochastic Pull-back Metric to weight distances
    print("Processing logic hops with Stochastic Riemannian Metric...")
    assorted_samples = create_assorted_dataset_riemannian(
        vae_model=vae,
        llm_tokenizer=tokenizer,
        data_list=data_list,
        device=device,
        codebook_centroids=centroids,
        max_hops_to_replace=3 # Randomized abstraction depth
    )

    # 5. Save for Stage 2 LLM Training
    output_dir = os.path.dirname(PATH_PROCESSED_DATA)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving Riemannian assorted data to {PATH_PROCESSED_DATA}...")
    with open(PATH_PROCESSED_DATA, 'w') as f:
        for item in tqdm(assorted_samples, desc="Writing JSONL"):
            f.write(json.dumps(item) + '\n')

    print("--- ✅ Riemannian Preprocessing Complete ---")

if __name__ == "__main__":
    main()
