"""
Main training script for both Stage 1 (VQ-VAE) and Stage 2 (LLM).
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

# Import from our source files
from src.utils import (
    get_llm_tokenizer, MAX_SEQ_LEN, LLM_MODEL_NAME, 
    PATH_VQVAE_MODEL, PATH_LLM_MODEL, PATH_PROCESSED_DATA
)
from src.dataset import VQVAE_Dataset, AssortedDataset
from src.model.vae import VQVAEModel
from src.model.transformer import get_llm_model

def train_vqvae(
    num_epochs: int = 3, 
    batch_size: int = 16, 
    lr: float = 1e-4, 
    d_model: int = 256,
    num_embeddings: int = 1024
):
    """Runs the Stage 1 VQ-VAE training."""
    print("--- 🚀 Starting Stage 1: VQ-VAE Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load tokenizer
    tokenizer = get_llm_tokenizer()
    vocab_size = len(tokenizer)

    # 2. Load data
    print("Loading raw GSM8K data...")
    gsm8k_data = load_dataset("gsm8k", "main")['train']
    train_dataset = VQVAE_Dataset(tokenizer, gsm8k_data, max_length=MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(train_dataset)} samples.")

    # 3. Initialize VQ-VAE model
    model = VQVAEModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_embeddings=num_embeddings,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 4. Training Loop
    model.train()
    for epoch in range(num_epochs):
        total_loss, total_recon, total_vq = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            optimizer.zero_grad()
            
            loss, recon_loss, vq_loss = model(input_ids)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_vq = total_vq / len(train_loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f}")

    # 5. Save the model
    print(f"Saving VQ-VAE model to {PATH_VQVAE_MODEL}")
    torch.save(model.state_dict(), PATH_VQVAE_MODEL)
    print("--- ✅ VQ-VAE Training Complete ---")

def train_llm(
    num_epochs: int = 1, 
    batch_size: int = 4, 
    lr: float = 2e-5
):
    """Runs the Stage 2 LLM fine-tuning."""
    print("--- 🚀 Starting Stage 2: LLM Fine-Tuning ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load tokenizer (with latent tokens)
    tokenizer = get_llm_tokenizer()

    # 2. Load the "assorted" dataset
    try:
        train_dataset = AssortedDataset(
            tokenizer, 
            file_path=PATH_PROCESSED_DATA, 
            max_length=MAX_SEQ_LEN
        )
    except FileNotFoundError:
        print(f"Error: Processed data not found at {PATH_PROCESSED_DATA}")
        print("Please run the 'notebooks/preprocessing.ipynb' notebook first.")
        return
    print(f"Loaded {len(train_dataset)} assorted samples.")

    # 3. Load the LLM and resize embeddings
    model = get_llm_model(LLM_MODEL_NAME, tokenizer_len=len(tokenizer))
    
    # 4. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=PATH_LLM_MODEL,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=50,
        save_steps=200,
        report_to="none", # Disable wandb/tensorboard
        fp16=True if device == "cuda" else False,
        push_to_hub=False,
    )

    # 5. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 6. Start Fine-Tuning
    print("Starting LLM fine-tuning...")
    trainer.train()
    
    # 7. Save the final model
    print(f"Saving final LLM to {PATH_LLM_MODEL}")
    trainer.save_model(PATH_LLM_MODEL)
    tokenizer.save_pretrained(PATH_LLM_MODEL)
    print("--- ✅ LLM Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Token Assorted models.")
    parser.add_argument(
        "stage", 
        type=str, 
        choices=["vqvae", "llm"], 
        help="Which model stage to train: 'vqvae' (Stage 1) or 'llm' (Stage 2)."
    )
    args = parser.parse_args()
    
    if args.stage == "vqvae":
        # You can add more argparse args for epochs, lr, etc.
        train_vqvae(num_epochs=3)
    elif args.stage == "llm":
        train_llm(num_epochs=1)