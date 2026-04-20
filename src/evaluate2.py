"""
Evaluates the fine-tuned LLM (Stage 2) on the GSM8K test set.
"""
import os
import re
import json
import torch
import shutil
from tqdm.auto import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple

# --- CONFIGURATION ---
# Replace these with your actual paths/names
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct" 
PATH_LLM_MODEL_ODDITY = "/content/drive/MyDrive/DLAI/experiments/llm_latent_oddity_finetune/final_model"
SAVE_DIR = PATH_LLM_MODEL_ODDITY
PRIMARY_FILE = os.path.join(SAVE_DIR, "eval_results.json")
BACKUP_FILE = os.path.join(SAVE_DIR, "eval_results_backup.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.utils import (
    get_llm_tokenizer, 
    parse_gsm8k_sample, 
    #parse_sample,
    extract_final_answer,
    PATH_LLM_MODEL
)

def evaluate_model(model_path):
    # --- 1. Setup Paths & Device ---
    save_dir = "/content/drive/MyDrive/DLAI/experiments/llm_latent_oddity_finetune/evaluate/"
    primary_file = os.path.join(save_dir, "eval_results.json")
    backup_file = os.path.join(save_dir, "eval_results_backup.json")
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct" 

    print(f"--- 📊 Evaluating Model from {model_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. Load Checkpoint / Progress ---
    correct = 0
    total = 0
    start_idx = 0
    
    load_path = None
    if os.path.exists(primary_file):
        load_path = primary_file
    elif os.path.exists(backup_file):
        load_path = backup_file
        print("⚠️ Primary checkpoint missing. Loading from Backup.")

    if load_path:
        try:
            with open(load_path, "r") as f:
                ckpt = json.load(f)
                correct = ckpt.get("correct", 0)
                total = ckpt.get("total", 0)
                start_idx = total
                print(f"🔄 Resuming from sample {start_idx}...")
        except Exception as e:
            print(f"❌ Could not load checkpoint ({e}). Starting fresh.")

    # --- 3. Load Model & Tokenizer ---
    print(f"Loading Base Model: {base_model_id}")
    tokenizer = get_llm_tokenizer()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # --- 4. Load & Slice Test Data ---
    full_test_data = load_dataset("gsm8k", "main")['test']
    if start_idx >= len(full_test_data):
        print("✅ Evaluation already complete.")
        return
        
    # Only evaluate what is left
    test_data = full_test_data.select(range(start_idx, len(full_test_data)))
    print(f"Loaded {len(full_test_data)} test samples. Remaining to process: {len(test_data)}.")
    
    # Initialize pbar with 'initial' set to start_idx for accurate progress display
    pbar = tqdm(test_data, desc="Evaluating", initial=start_idx, total=len(full_test_data))
    
    for i, sample in enumerate(pbar):
        # Current index in the context of the FULL dataset
        current_global_idx = start_idx + i
        
        parsed = parse_gsm8k_sample(sample)
        if not parsed:
            pbar.write(f"Skipping sample {current_global_idx}: Failed to parse.")
            total += 1 # Count skipped as processed
            continue
        
        prompt = parsed['prompt']
        solution = parsed['solution']
        true_answer = extract_final_answer(solution)
        
        if true_answer is None:
            pbar.write(f"Skipping sample {current_global_idx}: No true answer found.")
            total += 1
            continue
            
        # 5. Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred_answer = extract_final_answer(generated_text)
        
        # 6. Check for correctness
        if pred_answer is not None and pred_answer == true_answer:
            correct += 1
        total += 1

        # Live Stats
        current_acc = (correct / total) * 100 if total > 0 else 0
        pbar.set_postfix({"acc": f"{current_acc:.2f}%", "correct": correct, "total": total})

        # --- 7. Rolling Checkpoint (Save every 50 samples) ---
        if total % 50 == 0:
            # Shift primary to backup
            if os.path.exists(primary_file):
                shutil.copy2(primary_file, backup_file)
            
            # Save current state to primary
            results = {
                "accuracy": current_acc,
                "correct": correct,
                "total": total,
                "status": "in_progress"
            }
            with open(primary_file, "w") as f:
                json.dump(results, f, indent=4)

    # --- 8. Final Report & Save ---
    accuracy = (correct / total) * 100 if total > 0 else 0
    print("\n" + "="*30)
    print("📈 FINAL EVALUATION RESULTS")
    print(f"Correct: {correct} | Total: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*30)
    
    final_results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "status": "completed"
    }

    # Rotate one last time before final save
    if os.path.exists(primary_file):
        shutil.copy2(primary_file, backup_file)
        
    with open(primary_file, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"✅ Final results saved to {primary_file}")
