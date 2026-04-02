"""
Main training script for ProntoQA.
Stage 1: VQ-VAE (Replication of Token Assorted)
Stage 2: LLM (GPT-2) Fine-Tuning
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

# Import updated utilities and constants
from src.utils import (
    get_llm_tokenizer, MAX_SEQ_LEN, LLM_MODEL_NAME, 
    PATH_VQVAE_MODEL, PATH_VAE_MODEL, PATH_LLM_MODEL, 
    PATH_PROCESSED_DATA, PATH_RAW_DATA
)
from src.dataset import Lazy_ProntoQA_VQVAE_Dataset, AssortedDataset
from src.model.vae import VQVAEModel # Standard VQ-VAE for Stage 1
from src.model.transformer import get_llm_model

def train_vqvae(
    num_epochs: int = 5, 
    batch_size: int = 32, 
    lr: float = 1e-4, 
    num_embeddings: int = 1024,
    embedding_dim: int = 256
):
    """
    Runs Stage 1 Replication: Training a standard VQ-VAE on ProntoQA reasoning steps.
    This mimics the discrete codebook approach from the Token Assorted paper.
    """
    print("--- 🚀 Stage 1: VQ-VAE Training (Token Assorted Replication) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Tokenizer
    tokenizer = get_llm_tokenizer()
    vocab_size = len(tokenizer)

    # 2. Load ProntoQA reasoning hops
    train_dataset = Lazy_ProntoQA_VQVAE_Dataset(
        tokenizer, 
        file_path=PATH_RAW_DATA, 
        max_length=128
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(train_dataset)} logic hops for codebook learning.")

    # 3. Initialize VQ-VAE
    # In VQ-VAE, discretization happens DURING training via the vector quantizer layer.
    model = VQVAEModel(
        vocab_size=vocab_size,
        d_model=embedding_dim,
        num_embeddings=num_embeddings,
        max_seq_len=128
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    model.train()
    for epoch in range(num_epochs):
        total_loss, total_recon, total_vq = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"VQ-VAE Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            optimizer.zero_grad()
            
            # Loss = Reconstruction Loss + VQ Commitment Loss
            loss, recon_loss, vq_loss = model(input_ids)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Total Loss: {avg_loss:.4f} | Recon: {total_recon/len(train_loader):.4f} | VQ: {total_vq/len(train_loader):.4f}")

    # 5. Save the discrete VQ-VAE
    print(f"Saving VQ-VAE model to {PATH_VQVAE_MODEL}")
    torch.save(model.state_dict(), PATH_VQVAE_MODEL)
    print("--- ✅ VQ-VAE Training Complete ---")

def train_llm(
    num_epochs: int = 3, 
    batch_size: int = 8, 
    lr: float = 5e-5
):
    """
    Runs Stage 2: GPT-2 Fine-Tuning.
    This works for BOTH your Stage 1 replication and Stage 2 innovation,
    as long as the 'assorted' data file is formatted correctly.
    """
    print("--- 🚀 Stage 2: LLM Fine-Tuning (Assorted Reasoner) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_llm_tokenizer()

    # Load the processed dataset (with substituted <latent_i> tokens)
    try:
        train_dataset = AssortedDataset(
            tokenizer, 
            file_path=PATH_PROCESSED_DATA, 
            max_length=MAX_SEQ_LEN
        )
    except FileNotFoundError:
        print(f"Error: Processed data not found at {PATH_PROCESSED_DATA}")
        return

    # Load GPT-2 with expanded embeddings
    model = get_llm_model(LLM_MODEL_NAME, tokenizer_len=len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir=PATH_LLM_MODEL,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="epoch",
        fp16=True if device.type == "cuda" else False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Training GPT-2 to reason using Latent + Text tokens...")
    trainer.train()
    
    trainer.save_model(PATH_LLM_MODEL)
    tokenizer.save_pretrained(PATH_LLM_MODEL)
    print("--- ✅ LLM Fine-Tuning Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Token Assorted models on ProntoQA.")
    parser.add_argument(
        "stage", 
        type=str, 
        choices=["vqvae", "llm"], 
        help="Train Stage 1 (vqvae) or Stage 2 (llm)."
    )
    args = parser.parse_args()
    
    if args.stage == "vqvae":
        train_vqvae(num_epochs=5)
    elif args.stage == "llm":
        train_llm(num_epochs=3)
    
    if args.stage == "vqvae":
        # You can add more argparse args for epochs, lr, etc.
        train_vqvae(num_epochs=3)
    elif args.stage == "llm":
        train_llm(num_epochs=1)
