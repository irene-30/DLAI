"""
Evaluates the fine-tuned LLM (Stage 2) on the GSM8K test set.
"""
import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from src.utils import (
    get_llm_tokenizer, 
    #parse_gsm8k_sample, 
    parse_sample,
    extract_final_answer,
    PATH_LLM_MODEL
)

from peft import PeftModel
from transformers import AutoTokenizer

def evaluate_model(model_path):
    # 1. Load the original base model name (e.g., "meta-llama/Llama-3-8B")
    # This must match LLM_MODEL_NAME from your training
    base_model_id = "YOUR_BASE_MODEL_NAME_HERE" 
    
    print(f"Loading Base Model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # 2. Resize embeddings to match your trained tokenizer
    tokenizer = get_llm_tokenizer()
    base_model.resize_token_embeddings(len(tokenizer))
    
    # 3. Load the LoRA adapter weights from your Drive path
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # 4. Merge if you want better inference speed (optional)
    # model = model.merge_and_unload()
    
    model.eval()
    # ... rest of your evaluation logic
