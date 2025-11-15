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
    This version now pre-loads all text into a Python list
    to prevent deadlocks with num_workers.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 1. Load the HF dataset (this is fast, it's memory-mapped)
        print(f"Loading dataset from disk: {file_path}...")
        hf_dataset = load_dataset("json", data_files=file_path, split="train")
        
        # 2. (THE FIX) Extract all text into a plain Python list.
        #    This forces all text into RAM and breaks the link to the
        #    'datasets' library object, which solves the num_workers deadlock.
        print("Copying all text samples into RAM (this may take a moment)...")
        self.text_samples = [sample['text'] for sample in tqdm(hf_dataset)]
        print(f"Copied {len(self.text_samples)} samples.")
        
        # We no longer need the 'datasets' object
        del hf_dataset

    def __len__(self):
        # Return the length of our new list
        return len(self.text_samples)

    def __getitem__(self, idx):
        # 3. Get text from the simple Python list (very fast)
        text = self.text_samples[idx]
        
        # 4. Tokenize the text (this is the slow part that will now be
        #    parallelized by the DataLoader's workers)
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Remove the batch dimension (e.g. [1, 512] -> [512])
        # The DataLoader will re-add it.
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["labels"].squeeze(0)
        }
