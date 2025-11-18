"""
Step 1 Script: Train the Continuous VAE.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from src.utils import get_llm_tokenizer, MAX_SEQ_LEN, PATH_RAW_TRAIN
from src.dataset import Lazy_VQVAE_Dataset
from src.model.vae_continuous import ContinuousVAE

# Define path for the continuous model
PATH_CONTINUOUS_VAE = "experiments/vae_continuous.pth"

def train_continuous_vae(num_epochs=10, batch_size=16, lr=1e-4):
    print("--- 🚀 Step 1: Training Continuous VAE ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data
    tokenizer = get_llm_tokenizer()
    if os.path.exists(PATH_RAW_TRAIN):
        raw_dataset = load_dataset("json", data_files=PATH_RAW_TRAIN, split="train")
    else:
        raw_dataset = load_dataset("gsm8k", "main")['train']
        
    train_dataset = Lazy_VQVAE_Dataset(tokenizer, raw_dataset, max_length=MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 2. Model
    model = ContinuousVAE(
        vocab_size=len(tokenizer),
        d_model=256,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Train
    model.train()
    for epoch in range(num_epochs):
        total_loss, total_recon, total_kl = 0, 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            optimizer.zero_grad()
            
            loss, recon, kl = model(input_ids)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            
            progress.set_description(f"Loss: {loss.item():.4f} | KL: {kl.item():.5f}")

    # 4. Save
    os.makedirs(os.path.dirname(PATH_CONTINUOUS_VAE), exist_ok=True)
    torch.save(model.state_dict(), PATH_CONTINUOUS_VAE)
    print(f"Continuous VAE saved to {PATH_CONTINUOUS_VAE}")

if __name__ == "__main__":
    train_continuous_vae()
