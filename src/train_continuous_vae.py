"""
Step 1 (Stage 2 Innovation): Train the Continuous VAE on ProntoQA reasoning hops.
This provides the smooth latent space required for Riemannian post-hoc discretization.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import updated utilities and datasets for ProntoQA
from src.utils import (
    get_llm_tokenizer, 
    PATH_RAW_DATA,     # Path to your generated prontoqa_5hop.json
    PATH_VAE_MODEL     # "experiments/vae_continuous.pth"
)
from src.dataset import Lazy_ProntoQA_VAE_Dataset
from src.model.vae_continuous import ContinuousVAE

def train_continuous_vae(num_epochs=10, batch_size=32, lr=1e-4, latent_dim=128):
    print("--- 🚀 Stage 2: Training Continuous VAE (Riemannian Foundation) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tokenizer and Data
    tokenizer = get_llm_tokenizer()
    
    # Use the ProntoQA-specific dataset loader
    # This class extracts individual hops from the JSON file
    if not os.path.exists(PATH_RAW_DATA):
        print(f"❌ Error: ProntoQA data not found at {PATH_RAW_DATA}")
        print("Please run the 'run_experiment.py' script from the prontoqa repo first.")
        return

    train_dataset = Lazy_ProntoQA_VAE_Dataset(
        tokenizer, 
        file_path=PATH_RAW_DATA, 
        max_length=128 # Hops in ProntoQA are generally short sentences
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    print(f"Successfully loaded {len(train_dataset)} logic hops for training.")

    # 2. Continuous VAE Model
    # Note: Ensure ContinuousVAE returns (loss, recon_loss, kl_loss)
    model = ContinuousVAE(
        vocab_size=len(tokenizer),
        d_model=256,
        latent_dim=latent_dim,
        max_seq_len=128
    ).to(device)
    
    # We use a slightly lower weight for KL initially to prevent posterior collapse
    # or implement a KL-annealing strategy if the recon loss stays high.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    model.train()
    for epoch in range(num_epochs):
        total_loss, total_recon, total_kl = 0, 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            optimizer.zero_grad()
            
            # Forward pass: 
            # In Stage 2, the model learns a smooth Gaussian distribution
            loss, recon, kl = model(input_ids)
            
            loss.backward()
            
            # Gradient clipping is recommended for VAEs to prevent numerical instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            
            # Update tqdm progress bar with live metrics
            progress.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Recon": f"{recon.item():.4f}", 
                "KL": f"{kl.item():.5f}"
            })

    # 4. Save the Model
    # This .pth file will be the input for the Riemannian discretization script
    os.makedirs(os.path.dirname(PATH_VAE_MODEL), exist_ok=True)
    torch.save(model.state_dict(), PATH_VAE_MODEL)
    print(f"\n✅ Continuous VAE saved to {PATH_VAE_MODEL}")
    print("Next Step: Run Riemannian discretization to generate the Stage 2 Assorted Dataset.")

if __name__ == "__main__":
    train_continuous_vae()
