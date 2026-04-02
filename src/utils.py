"""
Utility functions, constants, and Riemannian data processing for ProntoQA experiments.
Implementing the "Latent-Oddity" (1710.11379) metric for Post-Hoc Discretization.
"""
import re
import json
import random
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- Constants ---
LLM_MODEL_NAME = "gpt2"
VQ_CODEBOOK_SIZE = 1024
MAX_SEQ_LEN = 512  # ProntoQA chains are shorter than GSM8K

# --- Token Definitions ---
SPECIAL_TOKENS = ["[PAD]", "[boLatent]", "[eoLatent]"]
LATENT_TOKENS = [f"<latent_{i}>" for i in range(VQ_CODEBOOK_SIZE)]

# --- Path Definitions ---
PATH_VAE_MODEL = "experiments/continuous_vae.pth"
PATH_LLM_MODEL = "experiments/llm_stage2"
PATH_RAW_DATA = "data/prontoqa_5hop.json" 
PATH_PROCESSED_DATA = "data/processed/assorted_riemannian.jsonl"

# --- 1. Tokenizer Setup ---

def get_llm_tokenizer() -> PreTrainedTokenizer:
    """Loads GPT-2 tokenizer and injects latent tokens into the vocabulary."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    tokenizer.add_special_tokens({
        'additional_special_tokens': SPECIAL_TOKENS[1:] + LATENT_TOKENS
    })
    return tokenizer

# --- 2. Data Parsing ---

def parse_prontoqa_sample(sample: Dict) -> Optional[Dict]:
    """Parses ProntoQA JSON into Question, Reasoning Hops, and Answer."""
    question = sample.get('question', '')
    cot_list = sample.get('chain_of_thought', [])
    answer = str(sample.get('answer', '')).lower()

    if not question or not cot_list:
        return None 

    return {
        'prompt': f"Question: {question}\nReasoning: ", 
        'cot_hops': cot_list,
        'solution': f" The answer is {answer}."
    }

# --- 3. Riemannian Metric Logic (arXiv:1710.11379) ---

def compute_stochastic_riemannian_weight(vae_model: torch.nn.Module, z: torch.Tensor, n_samples: int = 5) -> torch.Tensor:
    """
    Approximates the local 'speed' of the decoder at point z.
    This corresponds to the trace of the Pull-back Metric G(z).
    Points with high weights are 'oddities' where small Z changes cause large Text changes.
    """
    z = z.detach().clone().requires_grad_(True)
    
    # Enable gradients for the input latent vector
    with torch.set_grad_enabled(True):
        # Decode z to logits [Batch, Seq, Vocab]
        logits = vae_model.decode(z)
        
        jacobian_norm = 0
        for _ in range(n_samples):
            # Stochastic approximation using random projections
            v = torch.randn_like(logits)
            # Compute vector-jacobian product
            logits.backward(v, retain_graph=True)
            jacobian_norm += z.grad.norm(p=2)
            z.grad.zero_()
            
    return (jacobian_norm / n_samples).detach()

# --- 4. Dataset Creation (Assortment) ---

def create_assorted_dataset_riemannian(
    vae_model: torch.nn.Module,
    llm_tokenizer: PreTrainedTokenizer,
    data_list: List[Dict],
    device: torch.device,
    codebook_centroids: torch.Tensor, # Matrix of [1024, Latent_Dim]
    max_hops_to_replace: int = 3
) -> List[Dict[str, str]]:
    """
    Generates the 'Token Assorted' dataset using Riemannian-weighted discretization.
    """
    print(f"--- 📐 Generating Riemannian Assorted Dataset (K={VQ_CODEBOOK_SIZE}) ---")
    vae_model.to(device)
    vae_model.eval()
    
    assorted_samples = []
    boLatent, eoLatent = SPECIAL_TOKENS[1], SPECIAL_TOKENS[2]

    for sample in tqdm(data_list):
        parsed = parse_prontoqa_sample(sample)
        if not parsed: continue
        
        prompt, hops, solution = parsed['prompt'], parsed['cot_hops'], parsed['solution']
        
        # Decide how many hops to abstract (0 to 3)
        m = random.randint(0, min(len(hops), max_hops_to_replace))

        latent_strs = []
        if m > 0:
            for i in range(m):
                # Tokenize the specific logical hop
                hop_in = llm_tokenizer(hops[i], return_tensors="pt", truncation=True).to(device)
                
                with torch.no_grad():
                    # Get continuous latent representation (mu)
                    _, mu, _ = vae_model.encode(hop_in['input_ids'])
                
                # Compute the Riemannian weight to adjust the distance
                # This ensures we are more sensitive to boundaries in the latent space
                g_weight = compute_stochastic_riemannian_weight(vae_model, mu)
                
                # Calculate distances to all centroids in the codebook
                # Dist_R = ||mu - centroid|| * g_weight
                diffs = mu - codebook_centroids.to(device) # [1024, dim]
                euclidean_dist = torch.norm(diffs, dim=1)
                riemannian_dist = euclidean_dist * g_weight
                
                # Find the index of the functionally closest discrete token
                latent_idx = torch.argmin(riemannian_dist).item()
                latent_strs.append(LATENT_TOKENS[latent_idx])

            # Construct the reasoning string: [boL] <lat_1> <lat_2> [eoL] remaining_text_hops
            remaining_text = " ".join(hops[m:])
            assorted_cot = f"{boLatent} {' '.join(latent_strs)} {eoLatent} {remaining_text}"
        else:
            assorted_cot = " ".join(hops)

        # Final sequence for Stage 2 Training: P + C_assorted + S
        final_text = f"{prompt}{assorted_cot}{solution}"
        assorted_samples.append({"text": final_text})

    return assorted_samples

# --- 5. Helper for Evaluation ---

def extract_prontoqa_answer(text: str) -> Optional[str]:
    """Helper to find the final True/False prediction in model output."""
    normalized = text.lower().strip()
    # Priority 1: Direct mention of "the answer is..."
    match = re.search(r"the answer is (true|false|yes|no)", normalized)
    if match:
        val = match.group(1)
        return "true" if val in ["true", "yes"] else "false"
    
    # Priority 2: Last word check
    words = normalized.split()
    if not words: return None
    if words[-1].strip(".") in ["true", "yes"]: return "true"
    if words[-1].strip(".") in ["false", "no"]: return "false"
    
    return None
