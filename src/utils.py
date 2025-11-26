"""
Utility functions, constants, and data processing helpers.
Updated with robust parsing for MetaMathQA's varied formats.
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
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct" 
VQ_CODEBOOK_SIZE = 1024
MAX_SEQ_LEN = 1024 

# --- Token Definitions ---
SPECIAL_TOKENS = ["[PAD]", "[boLatent]", "[eoLatent]"] 
LATENT_TOKENS = [f"<latent_{i}>" for i in range(VQ_CODEBOOK_SIZE)]

# --- Smart Path Configuration ---
IN_COLAB = 'google.colab' in sys.modules
DRIVE_MOUNT_POINT = "/content/drive"
DRIVE_PATH = os.path.join(DRIVE_MOUNT_POINT, "My Drive")

if IN_COLAB and os.path.exists(DRIVE_PATH):
    PROJECT_SAVE_ROOT = os.path.join(DRIVE_PATH, "TokenAssorted_GSM8K_Results")
else:
    PROJECT_SAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Centralized Path Definitions ---
PATH_DATA_ROOT = os.path.join(PROJECT_SAVE_ROOT, "data")
PATH_EXPERIMENTS_ROOT = os.path.join(PROJECT_SAVE_ROOT, "experiments")

PATH_RAW_TRAIN = os.path.join(PATH_DATA_ROOT, "raw", "metamath_train.jsonl") 
PATH_RAW_TEST = os.path.join(PATH_DATA_ROOT, "raw", "gsm8k_test.jsonl")

PATH_PROCESSED_DATA = os.path.join(PATH_DATA_ROOT, "processed", "assorted_metamath_train.jsonl")
PATH_PROCESSED_DATA_RBF = os.path.join(PATH_DATA_ROOT, "processed", "assorted_metamath_train_rbf.jsonl")

PATH_VQVAE_MODEL = os.path.join(PATH_EXPERIMENTS_ROOT, "vqvae_metamath.pth")
PATH_LLM_MODEL = os.path.join(PATH_EXPERIMENTS_ROOT, "llama_stage2")
PATH_EVAL_RESULTS = os.path.join(PATH_EXPERIMENTS_ROOT, "evaluation_results.json")
# ------------------------------------

def get_llm_tokenizer() -> PreTrainedTokenizer:
    print(f"Loading tokenizer: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        'additional_special_tokens': SPECIAL_TOKENS + LATENT_TOKENS
    })
    return tokenizer

def parse_sample(sample: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Robust parser for GSM8K and MetaMathQA.
    Handles multiple answer delimiters (####, The answer is, etc.)
    """
    # 1. Extract Question and Answer Text
    if 'query' in sample and 'response' in sample:
        question = sample['query']
        answer_text = sample['response']
    elif 'question' in sample and 'answer' in sample:
        question = sample['question']
        answer_text = sample['answer']
    else:
        return None

    # 2. Try finding the answer using "####" (GSM8K style)
    match = re.search(r'####\s*(-?\d+[\.,\d]*)', answer_text)
    
    if match:
        final_answer_str = match.group(1).replace(',', '')
        # CoT is everything before ####
        cot_text = answer_text.split('####')[0].strip()
        
    else:
        # 3. Fallback: Try finding "The answer is" (MetaMath style)
        # Look for "The answer is" followed by a number at the end of the string
        # This regex looks for the number pattern near the end
        match_text = re.search(r'[Tt]he answer is[:\s]*(-?\d+[\.,\d]*)', answer_text)
        
        if match_text:
            final_answer_str = match_text.group(1).replace(',', '')
            # CoT is everything before "The answer is"
            # We split by the regex match to be safe
            cot_text = answer_text[:match_text.start()].strip()
        else:
            # 4. Final Fallback: Just grab the very last number in the text
            # This is risky but often necessary for messy datasets
            numbers = re.findall(r'-?\d+[\.,\d]*', answer_text)
            if numbers:
                final_answer_str = numbers[-1].replace(',', '')
                # We assume the whole text is CoT since we couldn't find a clear delimiter
                # But we remove the answer number from the end
                cot_text = answer_text.rstrip(final_answer_str).strip()
                # Clean up trailing punctuation or "The answer is" words
                cot_text = re.sub(r'[Tt]he answer is[:\s]*$', '', cot_text).strip()
            else:
                # Give up if absolutely no number found
                return None

    prompt = f"Question: {question}\nAnswer: "
    cot = cot_text
    solution = f" #### {final_answer_str}"
    
    return prompt, cot, solution

def extract_final_answer(text: str) -> Optional[str]:
    """Extracts numerical answer from model output."""
    # Try strict format first
    match = re.search(r'####\s*(-?\d+[\.,\d]*)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Try loose format (just the last number)
    numbers = re.findall(r'-?\d+[\.,\d]*', text)
    if numbers:
        return numbers[-1].replace(',', '')
        
    return None

# --- Preprocessing Functions (unchanged logic, just updated imports/signatures) ---

def create_assorted_dataset(
    vq_model, 
    llm_tokenizer: PreTrainedTokenizer, 
    dataset: HFDataset,
    device: torch.device,
    compression_rate: int = 16
) -> List[Dict[str, str]]:
    print("Creating 'Token Assorted' dataset (Standard VQ-VAE)...")
    vq_model.eval()
    assorted_samples = []
    boLatent = SPECIAL_TOKENS[1]
    eoLatent = SPECIAL_TOKENS[2]

    for sample in tqdm(dataset):
        parsed = parse_sample(sample)
        if not parsed: continue
        prompt, cot, solution = parsed

        cot_tokens = llm_tokenizer(cot, add_special_tokens=False, return_tensors="pt", max_length=MAX_SEQ_LEN, truncation=True)['input_ids']
        if cot_tokens.shape[1] == 0: continue
        
        with torch.no_grad():
            _, _, indices = vq_model.encode(cot_tokens.to(device))
        indices = indices.squeeze(0)

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

def create_assorted_dataset_rbf(
    vae_model,
    quantizer_model,
    llm_tokenizer: PreTrainedTokenizer,
    dataset: HFDataset,
    device: torch.device,
    compression_rate: int = 16
) -> List[Dict[str, str]]:
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

        cot_tokens = llm_tokenizer(cot, add_special_tokens=False, return_tensors="pt", max_length=MAX_SEQ_LEN, truncation=True)['input_ids']
        if cot_tokens.shape[1] == 0: continue
        
        input_ids = cot_tokens.to(device)
        
        with torch.no_grad():
            _, z_mu, _ = vae_model.encode(input_ids)
            indices = quantizer_model.get_indices(z_mu)
        indices = indices.squeeze(0)

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
