"""
Preprocessing Script for RBF Post-Hoc Model.
Loads Continuous VAE + RBF Quantizer -> Creates Assorted Dataset.
"""
import torch
import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Add 'src' to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from utils import (
    get_llm_tokenizer, MAX_SEQ_LEN, VQ_CODEBOOK_SIZE, 
    PATH_PROCESSED_DATA_RBF, create_assorted_dataset_rbf,
    PATH_RAW_TRAIN
)
from model.vae_continuous import ContinuousVAE
from model.quantizer_posthoc import PostHocRBFQuantizer

PATH_CONTINUOUS_VAE = os.path.join(project_root, "experiments/vae_continuous.pth")
PATH_RBF_QUANTIZER = os.path.join(project_root, "experiments/rbf_quantizer_posthoc.pth")

def main():
    print("--- 🚀 Starting RBF Assorted Dataset Preprocessing ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = get_llm_tokenizer()

    # 1. Load Continuous VAE
    print("Loading Continuous VAE...")
    vae = ContinuousVAE(len(tokenizer), d_model=256, max_seq_len=MAX_SEQ_LEN).to(device)
    vae.load_state_dict(torch.load(PATH_CONTINUOUS_VAE, map_location=device))
    
    # 2. Load RBF Quantizer
    print("Loading RBF Quantizer...")
    quantizer = PostHocRBFQuantizer(VQ_CODEBOOK_SIZE, embedding_dim=256, gamma=10.0).to(device)
    quantizer.load_state_dict(torch.load(PATH_RBF_QUANTIZER, map_location=device))

    # 3. Load Data
    print("Loading raw data...")
    if os.path.exists(PATH_RAW_TRAIN):
        raw_dataset = load_dataset("json", data_files=PATH_RAW_TRAIN, split="train")
    else:
        raw_dataset = load_dataset("gsm8k", "main")['train']

    # 4. Create Assorted Data
    assorted_samples = create_assorted_dataset_rbf(
        vae_model=vae,
        quantizer_model=quantizer,
        llm_tokenizer=tokenizer,
        dataset=raw_dataset,
        device=device
    )

    # 5. Save
    output_dir = os.path.dirname(PATH_PROCESSED_DATA_RBF)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {PATH_PROCESSED_DATA_RBF}...")
    
    with open(PATH_PROCESSED_DATA_RBF, 'w') as f:
        for item in tqdm(assorted_samples):
            f.write(json.dumps(item) + '\n')
    
    print("--- ✅ RBF Preprocessing Complete ---")

if __name__ == "__main__":
    main()
