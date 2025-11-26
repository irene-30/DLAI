"""
Utility functions, constants, and data processing helpers.
"""
import re
import json
import random
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.model.vae import VQVAEModel # Import for type hinting and usage
# Import models for type hinting
from src.model.vae_continuous import ContinuousVAE
from src.model.quantizer_posthoc import PostHocRBFQuantizer

# --- Constants ---
LLM_MODEL_NAME = "gpt2"
VQ_CODEBOOK_SIZE = 1024
MAX_SEQ_LEN = 1024

# --- Token Definitions ---
SPECIAL_TOKENS = ["[PAD]", "[boLatent]", "[eoLatent]"]
LATENT_TOKENS = [f"<latent_{i}>" for i in range(VQ_CODEBOOK_SIZE)]

# --- Path Definitions ---
# These paths are relative to the repository root
PATH_VQVAE_MODEL = "/content/drive/My Drive/DLAI/experiments/vqvae_stage1.pth"     # "experiments/vqvae_stage1.pth"
PATH_LLM_MODEL = "/content/drive/My Drive/DLAI/experiments/llm_stage2"     # "experiments/llm_stage2"
PATH_RAW_DATA = "data/raw/gsm8k_train.jsonl" # Assumes you downloaded it
PATH_PROCESSED_DATA = "/content/drive/My Drive/DLAI/data/processed/assorted_train.jsonl"   # "data/processed/assorted_train.jsonl"
# PATH_PROCESSED_DATA_RBF = "/content/drive/My Drive/DLAI/data/processed/assorted_train_rbf.jsonl"


def get_llm_tokenizer() -> PreTrainedTokenizer:
    """
    Loads the main LLM tokenizer and adds all our new special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    
    # Add special tokens
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'additional_special_tokens': SPECIAL_TOKENS[1:] + LATENT_TOKENS
    })
    return tokenizer

def parse_gsm8k_sample(sample: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Splits a GSM8K sample into Prompt (P), CoT (C), and Solution (S).
    """
    question = sample.get('question', '')
    answer_text = sample.get('answer', '')
    
    match = re.search(r'####\s*(-?\d+[\.,\d]*)', answer_text)
    if not match:
        return None # Skip samples we can't parse

    final_answer_str = match.group(1).replace(',', '')
    cot_text = answer_text.split('####')[0].strip()
    
    prompt = f"Question: {question}\nAnswer: "
    cot = cot_text
    solution = f" #### {final_answer_str}"
    
    return {'prompt': prompt, 'cot':  cot, 'solution': solution}

def extract_final_answer(text: str) -> Optional[str]:
    """Extracts the final numerical answer from a generated string."""
    match = re.search(r'####\s*(-?\d+[\.,\d]*)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

def create_assorted_dataset(
    vq_model: VQVAEModel, 
    llm_tokenizer: PreTrainedTokenizer, 
    dataset: HFDataset,
    device: torch.device,
    compression_rate: int = 16, # As per paper
    max_latent_tokens: int = 256 # As per paper (randomized max)
) -> List[Dict[str, str]]:
    """
    Generates the "Token Assorted" dataset by mixing latent and text tokens.
    This is the core logic from the paper, run by the preprocessing notebook.
    
    Args:
        vq_model: The trained VQ-VAE model (Stage 1).
        llm_tokenizer: The tokenizer (with latent tokens added).
        dataset: The raw Hugging Face dataset.
        compression_rate: How many text tokens each latent token represents.
        max_latent_tokens: The max number of latent tokens to use per sample.
        
    Returns:
        A list of dictionaries, where each dict has a "text" key
        containing the final P + C_assorted + S string.
    """
    print("Creating 'Token Assorted' dataset...")
    vq_model.eval()
    assorted_samples = []
    
    # Pre-calculate replacement token IDs
    boLatent_token = SPECIAL_TOKENS[1]
    eoLatent_token = SPECIAL_TOKENS[2]

    for sample in tqdm(dataset):
        parsed = parse_gsm8k_sample(sample)
        if not parsed:
            continue
            
        prompt, cot, solution = parsed['prompt'], parsed['cot'], parsed['solution']

        # --- 1. Get Latent Tokens for the CoT (C) ---
        cot_tokens_dict = llm_tokenizer(
            cot, 
            add_special_tokens=False, 
            return_tensors="pt", 
            max_length=MAX_SEQ_LEN, 
            truncation=True
        )
        cot_tokens = cot_tokens_dict['input_ids']
        
        if cot_tokens.shape[1] == 0:
            continue
        
        with torch.no_grad():
            _, _, indices = vq_model.encode(cot_tokens.to(device))
        
        indices = indices.squeeze(0) # Shape (T_cot,)

        # --- 2. Apply Randomized Replacement (as per paper) ---
        
        # Randomly vary the number of text tokens to be substituted
        # m_max is randomly sampled from a set
        m_max_options = [0, 72, 128, 160, 192, 224, 256] 
        m_max = random.choice(m_max_options)
        
        # Sample m from [0, 16, ..., m_max]
        if m_max > 0:
            m = random.choice(range(0, m_max + 1, compression_rate))
        else:
            m = 0
            
        # Ensure m is not larger than the actual CoT token length
        m = min(m, cot_tokens.shape[1])

        if m > 0:
            # We replace the first 'm' text tokens.
            # These 'm' text tokens correspond to 'm' latent tokens.
            # (Note: The paper implies chunks. T=T_cot for simplicity here).
            num_latent_to_use = m
            
            # Get the latent token strings
            latent_token_strings = [
                LATENT_TOKENS[idx.item()] for idx in indices[:num_latent_to_use]
            ]
            
            # Get the remaining *un-tokenized* text
            # This is a simple approximation.
            split_point = cot_tokens_dict.token_to_chars(m - 1).end if m > 0 else 0
            remaining_cot_text = cot[split_point:]

            assorted_cot = (
                f"{boLatent_token} " + 
                " ".join(latent_token_strings) + 
                f" {eoLatent_token}" +
                remaining_cot_text
            )
        else:
            # m=0, use the full text CoT
            assorted_cot = cot

        # --- 3. Create the final training string (P + C_assorted + S) ---
        final_string = prompt + assorted_cot + solution
        assorted_samples.append({"text": final_string})

    return assorted_samples

def create_assorted_dataset_rbf(
    vae_model: ContinuousVAE,
    quantizer_model: PostHocRBFQuantizer,
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
    # RBF Quantizer does not have batch norm/dropout, but good practice
    quantizer_model.eval() 
    
    assorted_samples = []
    boLatent, eoLatent = SPECIAL_TOKENS[1], SPECIAL_TOKENS[2]

    for sample in tqdm(dataset):
        parsed = parse_gsm8k_sample(sample)
        if not parsed: continue
        prompt, cot, solution = parsed['prompt'], parsed['cot'], parsed['solution']

        cot_tokens = llm_tokenizer(cot, add_special_tokens=False, return_tensors="pt", max_length=MAX_SEQ_LEN, truncation=True)['input_ids']
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
            # Careful with offset mapping here if tokenizer is fast
            try:
                encoding = llm_tokenizer(cot, add_special_tokens=False)
                split_point = encoding.token_to_chars(m - 1).end
                remaining_text = cot[split_point:]
            except:
                 # Fallback if alignment fails slightly
                remaining_text = "" 
            
            assorted_cot = f"{boLatent} {' '.join(latent_strs)} {eoLatent}{remaining_text}"
        else:
            assorted_cot = cot

        assorted_samples.append({"text": prompt + assorted_cot + solution})

    return assorted_samples
