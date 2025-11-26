"""
Utility functions, constants, and data processing helpers.
Updated for Llama 3.2, QLoRA, and MetaMathQA dataset support.
"""
import os
import sys
import re
import json
import random
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- Constants ---
# Updated to Llama 3.2-3B
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct" 
VQ_CODEBOOK_SIZE = 1024
# Llama supports up to 128k, but we limit to 1024 or 512 to save VRAM during training
MAX_SEQ_LEN = 1024 

# --- Token Definitions ---
SPECIAL_TOKENS = ["[PAD]", "[boLatent]", "[eoLatent]"] 
LATENT_TOKENS = [f"<latent_{i}>" for i in range(VQ_CODEBOOK_SIZE)]

# --- Smart Path Configuration ---
# This automatically detects if you are in Colab to set the correct drive path
IN_COLAB = 'google.colab' in sys.modules
DRIVE_MOUNT_POINT = "/content/drive"
DRIVE_PATH = os.path.join(DRIVE_MOUNT_POINT, "My Drive")

if IN_COLAB and os.path.exists(DRIVE_PATH):
    # Saves to 'TokenAssorted_GSM8K_Results' in your Drive
    PROJECT_SAVE_ROOT = os.path.join(DRIVE_PATH, "TokenAssorted_GSM8K_Results")
else:
    # Saves locally if not in Colab
    PROJECT_SAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Centralized Path Definitions ---
PATH_DATA_ROOT = os.path.join(PROJECT_SAVE_ROOT, "data")
PATH_EXPERIMENTS_ROOT = os.path.join(PROJECT_SAVE_ROOT, "experiments")

# Raw Data Paths (Updated for MetaMathQA)
PATH_RAW_TRAIN = os.path.join(PATH_DATA_ROOT, "raw", "metamath_train.jsonl") 
PATH_RAW_TEST = os.path.join(PATH_DATA_ROOT, "raw", "gsm8k_test.jsonl")

# Processed Data Paths
PATH_PROCESSED_DATA = os.path.join(PATH_DATA_ROOT, "processed", "assorted_metamath_train.jsonl")
PATH_PROCESSED_DATA_RBF = os.path.join(PATH_DATA_ROOT, "processed", "assorted_metamath_train_rbf.jsonl")

# Model Checkpoint Paths
PATH_VQVAE_MODEL = os.path.join(PATH_EXPERIMENTS_ROOT, "vqvae_metamath.pth")
PATH_LLM_MODEL = os.path.join(PATH_EXPERIMENTS_ROOT, "llama_stage2")
PATH_EVAL_RESULTS = os.path.join(PATH_EXPERIMENTS_ROOT, "evaluation_results.json")
# ------------------------------------

def get_llm_tokenizer() -> PreTrainedTokenizer:
    """
    Loads the Llama 3.2 tokenizer and adds special latent tokens.
    Handles the missing PAD token issue for Llama.
    """
    print(f"Loading tokenizer: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    
    # Llama 3 does not have a default pad token. 
    # We explicitly set it to eos_token to avoid errors during batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Add our special tokens (latent tokens + boundary markers)
    tokenizer.add_special_tokens({
        'additional_special_tokens': SPECIAL_TOKENS + LATENT_TOKENS
    })
    return tokenizer

def parse_sample(sample: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Generic parser that handles BOTH GSM8K and MetaMathQA formats.
    
    Returns:
        (Prompt, CoT, Solution) tuple, or None if parsing fails.
    """
    # 1. Try MetaMathQA format (query / response)
    if 'query' in sample and 'response' in sample:
        question = sample['query']
        answer_text = sample['response']
    # 2. Try GSM8K format (question / answer)
    elif 'question' in sample and 'answer' in sample:
        question = sample['question']
        answer_text = sample['answer']
    # 3. Fallback for 'text' based datasets (if already processed)
    elif 'text' in sample:
        return None
    else:
        return None

    # Extract final answer (works for both datasets usually)
    # Both often use "#### 123" format
    match = re.search(r'####\s*(-?\d+[\.,\d]*)', answer_text)
    
    # If no delimiter found, we might skip to ensure we have clean CoT
    if not match:
        return None 

    final_answer_str = match.group(1).replace(',', '')
    
    # The CoT is everything before the final answer delimiter
    cot_text = answer_text.split('####')[0].strip()
    
    # Construct standard Prompt (P)
    prompt = f"Question: {question}\nAnswer: "
    # CoT (C)
    cot = cot_text
    # Solution (S)
    solution = f" #### {final_answer_str}"
    
    return prompt, cot, solution

def extract_final_answer(text: str) -> Optional[str]:
    """Extracts numerical answer from model output."""
    match = re.search(r'####\s*(-?\d+[\.,\d]*)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

# --- Preprocessing Functions ---

def create_assorted_dataset(
    vq_model, 
    llm_tokenizer: PreTrainedTokenizer, 
    dataset: HFDataset,
    device: torch.device,
    compression_rate: int = 16
) -> List[Dict[str, str]]:
    """
    Standard VQ-VAE preprocessing (Legacy/Stage 1).
    Encodes CoT using VQ-VAE indices.
    """
    print("Creating 'Token Assorted' dataset (Standard VQ-VAE)...")
    vq_model.eval()
    assorted_samples = []
    
    # Get special tokens
    boLatent = SPECIAL_TOKENS[1] # [boLatent]
    eoLatent = SPECIAL_TOKENS[2] # [eoLatent]

    for sample in tqdm(dataset):
        parsed = parse_sample(sample)
        if not parsed: continue
        prompt, cot, solution = parsed

        cot_tokens = llm_tokenizer(
            cot, 
            add_special_tokens=False, 
            return_tensors="pt", 
            max_length=MAX_SEQ_LEN, 
            truncation=True
        )['input_ids']
        
        if cot_tokens.shape[1] == 0: continue
        
        with torch.no_grad():
            # Encode to discrete indices
            _, _, indices = vq_model.encode(cot_tokens.to(device))
        indices = indices.squeeze(0)

        # Randomized Replacement Logic (as per paper)
        m_max_options = [0, 72, 128, 160, 192, 224, 256]
        m_max = random.choice(m_max_options)
        
        if m_max > 0:
            m = random.choice(range(0, m_max + 1, compression_rate))
        else:
            m = 0
        
        m = min(m, cot_tokens.shape[1])

        if m > 0:
            # Get latent tokens
            latent_strs = [LATENT_TOKENS[idx.item()] for idx in indices[:m]]
            
            try:
                encoding = llm_tokenizer(cot, add_special_tokens=False)
                split_point = encoding.token_to_chars(m - 1).end
                remaining_text = cot[split_point:]
            except:
                remaining_text = ""
            
            assorted_cot = f"{boLatent} {' '.join(latent_strs)} {eoLatent}{remaining_text}"
        else:
            assorted_cot = cot

        assorted_samples.append({"text": prompt + assorted_cot + solution})
    return assorted_samples

def create_assorted_dataset_rbf(
    vae_model,
    quantizer_model,
    llm_tokenizer: PreTrainedTokenizer,
    dataset: HFDataset,
    device: torch.device,
    compression_rate: int = 16
) -> List[Dict[str, str]]:
    """
    Generates the assorted dataset using the Two-Stage RBF approach.
    1. VAE encodes to continuous 'z'.
    2. RBF Quantizer maps 'z' to indices.
    """
    print("Creating 'Token Assorted' dataset (RBF Post-Hoc)...")
    vae_model.eval()
    quantizer_model.eval() 
    
    assorted_samples = []
    boLatent = SPECIAL_TOKENS[1]
    eoLatent = SPECIAL_TOKENS[2]

    for sample in tqdm(dataset):
        parsed = parse_sample(sample)
        if not parsed: continue
        prompt, cot, solution = parsed

        cot_tokens = llm_tokenizer(
            cot, 
            add_special_tokens=False, 
            return_tensors="pt", 
            max_length=MAX_SEQ_LEN, 
            truncation=True
        )['input_ids']
        
        if cot_tokens.shape[1] == 0: continue
        
        input_ids = cot_tokens.to(device)
        
        with torch.no_grad():
            # Step 1: Get continuous latent z (using mean)
            # ContinuousVAE.encode returns (z, mu, logvar)
            _, z_mu, _ = vae_model.encode(input_ids)
            
            # Step 2: Map continuous z to discrete indices using RBF
            indices = quantizer_model.get_indices(z_mu)
        
        indices = indices.squeeze(0)

        # --- Randomized Replacement (Same logic as standard) ---
        m_max = random.choice([0, 72, 128, 160, 192, 224, 256])
        m = random.choice(range(0, m_max + 1, compression_rate)) if m_max > 0 else 0
        m = min(m, cot_tokens.shape[1])

        if m > 0:
            latent_strs = [LATENT_TOKENS[idx.item()] for idx in indices[:m]]
            try:
                encoding = llm_tokenizer(cot, add_special_tokens=False)
                split_point = encoding.token_to_chars(m - 1).end
                remaining_text = cot[split_point:]
            except:
                remaining_text = "" 
            
            assorted_cot = f"{boLatent} {' '.join(latent_strs)} {eoLatent}{remaining_text}"
        else:
            assorted_cot = cot

        assorted_samples.append({"text": prompt + assorted_cot + solution})

    return assorted_samples
