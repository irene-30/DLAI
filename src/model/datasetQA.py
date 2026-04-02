import torch
import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import List, Dict

def parse_prontoqa_sample(sample: Dict) -> Dict:
    """
    Parses a ProntoQA sample into a structured format.
    ProntoQA usually provides 'question', 'chain_of_thought' (list), and 'answer'.
    """
    # Check if sample is valid
    if 'question' not in sample or 'chain_of_thought' not in sample:
        return None
    
    # In ProntoQA, CoT is often a list of strings representing each hop
    cot_steps = sample['chain_of_thought']
    if isinstance(cot_steps, list):
        cot_text = " ".join(cot_steps)
    else:
        cot_text = str(cot_steps)
        
    return {
        "prompt": sample['question'],
        "cot": cot_text,
        "solution": str(sample.get('answer', ''))
    }

class Lazy_ProntoQA_VQVAE_Dataset(Dataset):
    """
    Custom Dataset for Stage 1 (Training VAE).
    Maps ProntoQA reasoning hops into the latent space.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load the JSON/JSONL data
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # If ProntoQA is structured as a dict of examples, convert to list
        if isinstance(data, dict):
            # Some versions use "example_1", "example_2" keys
            data = list(data.values())

        for sample in data:
            parsed = parse_prontoqa_sample(sample)
            if not parsed:
                continue
            
            # For VAE training, we want the model to reconstruct the reasoning
            # We focus on the 'cot' (the hops) as that is what we are abstracting
            full_text = f"Context: {parsed['prompt']} Reasoning: {parsed['cot']}"
            self.samples.append(full_text)

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


class AssortedDataset(Dataset):
    """
    Custom Dataset for Stage 2 (Training LLM/GPT-2).
    Expects JSONL where 'text' contains the mixed Latent + Text tokens.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loading assorted ProntoQA dataset from {file_path}...")
        
        # Using Hugging Face 'datasets' for memory-mapping
        self.data = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The 'text' field should already have the <LATENT_i> tokens inserted
        # by your post-processing script.
        text = self.data[idx]['text']
        
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Optional: Mask labels for the prompt part so the model only 
        # learns to predict the Latents + CoT + Answer.
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels
        }
