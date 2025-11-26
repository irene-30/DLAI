"""
Step 2 Script: Post-Hoc Discretization.
1. Load trained Continuous VAE.
2. Freeze it.
3. Pass data through it to get 'z'.
4. Train RBF Quantizer on 'z'.
"""
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from src.utils import get_llm_tokenizer, MAX_SEQ_LEN, PATH_RAW_TRAIN, VQ_CODEBOOK_SIZE
from src.dataset import Lazy_VQVAE_Dataset
from src.model.vae_continuous import ContinuousVAE
from src.model.quantizer_posthoc import PostHocRBFQuantizer

PATH_CONTINUOUS_VAE = "experiments/vae_continuous.pth"
PATH_RBF_QUANTIZER = "experiments/rbf_quantizer_posthoc.pth"

def train_posthoc_quantizer(num_epochs=5, batch_size=32):
    print("--- 🚀 Step 2: Post-Hoc RBF Quantization ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    tokenizer = get_llm_tokenizer()
    if os.path.exists(PATH_RAW_TRAIN):
        raw_dataset = load_dataset("json", data_files=PATH_RAW_TRAIN, split="train")
    else:
        raw_dataset = load_dataset("gsm8k", "main")['train']
    
    train_dataset = Lazy_VQVAE_Dataset(tokenizer, raw_dataset, max_length=MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 2. Load & Freeze VAE
    vae = ContinuousVAE(
        vocab_size=len(tokenizer),
        d_model=256,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    
    try:
        vae.load_state_dict(torch.load(PATH_CONTINUOUS_VAE, map_location=device))
        print("Continuous VAE loaded.")
    except FileNotFoundError:
        print("Error: Train the Continuous VAE first!")
        return
    
    vae.eval() # Set to eval mode
    for param in vae.parameters():
        param.requires_grad = False # Freeze weights

    # 3. Initialize RBF Quantizer
    quantizer = PostHocRBFQuantizer(
        num_embeddings=VQ_CODEBOOK_SIZE,
        embedding_dim=256,
        gamma=10.0
    ).to(device)

    # 4. Train Quantizer (Clustering)
    # We just loop over data and update centroids
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} - Clustering Latent Space...")
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            
            with torch.no_grad():
                # Get continuous latent vectors
                # We use the mean (mu) as the representation to cluster
                _, mu, _ = vae.encode(input_ids)
            
            # Update RBF centroids
            quantizer.update(mu)

    # 5. Save Quantizer
    torch.save(quantizer.state_dict(), PATH_RBF_QUANTIZER)
    print(f"RBF Quantizer saved to {PATH_RBF_QUANTIZER}")

if __name__ == "__main__":
    train_posthoc_quantizer()
