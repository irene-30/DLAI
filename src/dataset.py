"""
Contains PyTorch Dataset classes for loading data.
Updated to support generic parsing for MetaMathQA and GSM8K.
"""
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import List, Dict
from src.utils import parse_sample  # Changed from parse_gsm8k_sample
from tqdm import tqdm

# -----------------------------------------------------------------
# VQVAE Dataset (Lazy Loading for Raw Data)
# -----------------------------------------------------------------
class Lazy_VQVAE_Dataset(Dataset):
    """
    Custom Dataset for Stage 1 (Training VQ-VAE).
    Parses raw JSON data (GSM8K or MetaMathQA) into full text sequences.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, data: List, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print("Parsing data into memory list...")
        for sample in tqdm(data, desc="Parsing samples"):
            # Use generic parser that handles both dataset formats
            parsed = parse_sample(sample)
            if not parsed:
                continue
            
            prompt, cot, solution = parsed
            # VQ-VAE trains on the full sequence: P + C + S
            full_text = prompt + cot + solution
            self.samples.append(full_text)
            
        print(f"Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_text = self.samples[idx]
        
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized['input_ids'].squeeze(0),
            "attention_mask": tokenized['attention_mask'].squeeze(0)
        }

# -----------------------------------------------------------------
# Assorted Dataset (Pre-loaded for Speed)
# -----------------------------------------------------------------
class AssortedDataset(Dataset):
    """
    Custom Dataset for Stage 2 (Training LLM).
    Pre-loads all text into a Python list to avoid deadlocks with num_workers.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading dataset from disk: {file_path}...")
        # Load the HF dataset (memory-mapped initially)
        hf_dataset = load_dataset("json", data_files=file_path, split="train")
        
        # CRITICAL SPEED FIX:
        # Extract all text into a plain Python list immediately.
        # This breaks the link to the 'datasets' object (Apache Arrow),
        # preventing deadlocks when using multiple workers in DataLoader.
        print("Copying all text samples into RAM (this may take a moment)...")
        self.text_samples = [sample['text'] for sample in tqdm(hf_dataset)]
        print(f"Copied {len(self.text_samples)} samples.")
        
        # Clean up the HF dataset object
        del hf_dataset

    def __len__(self):
        return len(self.text_samples)

    def __getitem__(self, idx):
        # 1. Get raw text (fast list access)
        text = self.text_samples[idx]
        
        # 2. Tokenize (CPU intensive - will be parallelized by workers)
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 3. Create labels for Causal LM training
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # 4. Remove batch dimension added by tokenizer
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["labels"].squeeze(0)
        }
