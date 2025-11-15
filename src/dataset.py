"""
Contains PyTorch Dataset classes for loading data.
"""
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import List, Dict
from src.utils import parse_gsm8k_sample


class Lazy_VQVAE_Dataset(Dataset):
    """
    Custom Dataset for Stage 1 (Training VQ-VAE).
    This "lazy" version tokenizes samples on-the-fly in __getitem__
    to keep RAM usage extremely low.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, data: List[Dict], max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # This is fast and uses almost no RAM.
        # We only store the *raw text strings*.
        for sample in data:
            parsed = parse_gsm8k_sample(sample)
            if not parsed:
                continue
            
            # Store the raw text, not the tokenized tensor
            full_text = parsed[0] + parsed[1] + parsed[2]
            self.samples.append(full_text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Tokenization happens here, one sample at a time.
        # This is extremely memory-efficient.
        full_text = self.samples[idx]
        
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Return the squeezed tensors
        return {
            "input_ids": tokenized['input_ids'].squeeze(0),
            "attention_mask": tokenized['attention_mask'].squeeze(0)
        }


class AssortedDataset(Dataset):
    """
    Custom Dataset for Stage 2 (Training LLM).
    Loads the *pre-processed* assorted dataset.
    This also uses a lazy-loading approach.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int):
        self.data = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"  # <-- 1. ADD THIS to return PyTorch Tensors
        )
        
        # 2. Use .clone() for labels
        tokenized["labels"] = tokenized["input_ids"].clone() 
        
        # 3. Squeeze all tensors to remove the batch dim (size [1, T] -> [T])
        # The DataLoader will re-add the batch dimension later.
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["labels"].squeeze(0)
        }
