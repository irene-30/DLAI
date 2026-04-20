"""
Evaluates the fine-tuned LLM (Stage 2) on the GSM8K test set.
"""
import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json

from src.utils import (
    get_llm_tokenizer, 
    parse_gsm8k_sample, 
    #parse_sample,
    extract_final_answer,
    PATH_LLM_MODEL
)

from peft import PeftModel
from transformers import AutoTokenizer

def evaluate_model(model_path):
    # 1. Load the original base model name
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct" 

    print(f"--- 📊 Evaluating Model from {model_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Base Model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # 2. Resize embeddings to match your trained tokenizer
    tokenizer = get_llm_tokenizer()
    base_model.resize_token_embeddings(len(tokenizer))
    
    # 3. Load the LoRA adapter weights
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # 4. Load test data
    test_data = load_dataset("gsm8k", "main")['test']
    print(f"Loaded {len(test_data)} test samples.")
    
    correct = 0
    total = 0
    
    # --- FIX 1: Assign tqdm to 'pbar' ---
    # We use enumerate(pbar) so we have an index 'i' for our print statements
    pbar = tqdm(test_data, desc="Evaluating")
    
    for i, sample in enumerate(pbar):
        parsed = parse_gsm8k_sample(sample)
        if not parsed:
            # Use pbar.write to avoid breaking the progress bar
            pbar.write(f"Skipping sample {i}: Failed to parse.")
            continue
        
        prompt = parsed['prompt']
        solution = parsed['solution']
        
        # Get ground truth answer
        true_answer = extract_final_answer(solution)
        if true_answer is None:
            pbar.write(f"Skipping sample {i}: Failed to extract true answer.")
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
        
        # 6. Check for correctness
        pred_answer = extract_final_answer(generated_text)
        
        if pred_answer is not None and pred_answer == true_answer:
            correct += 1
        total += 1

        # --- FIX 2: pbar now exists in this scope ---
        current_acc = (correct / total) * 100 if total > 0 else 0
        pbar.set_postfix({
            "acc": f"{current_acc:.2f}%", 
            "correct": correct,
            "total": total
        })

    # 7. Final Report
    accuracy = (correct / total) * 100 if total > 0 else 0
    print("\n--- 📈 Evaluation Results ---")
    print(f"Correct: {correct}")
    print(f"Total:   {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

    # 8. Save to Drive
    save_file = "/content/drive/MyDrive/DLAI/experiments/llm_latent_oddity_finetune/final_model/eval_results.json"
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to {save_file}")
