"""
Evaluates the fine-tuned LLM (Stage 2) on the ProntoQA test set.
"""
import torch
import json
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import re

# Assuming these are updated in your src/utils.py
from src.utils import (
    get_llm_tokenizer, 
    PATH_LLM_MODEL
)

def extract_prontoqa_answer(text: str) -> str:
    """
    Extracts the final True/False or Yes/No answer from the model's generated text.
    ProntoQA usually concludes with "The answer is [True/False]." or just "True."
    """
    text = text.lower().strip()
    # Search for common patterns at the end of the generation
    if "true" in text.split() or "yes" in text.split():
        return "true"
    if "false" in text.split() or "no" in text.split():
        return "false"
    
    # Fallback: find the last occurrence of true/false
    matches = re.findall(r'\b(true|false|yes|no)\b', text)
    if matches:
        final_match = matches[-1]
        return "true" if final_match in ["true", "yes"] else "false"
    
    return None

def evaluate_model(model_path: str = PATH_LLM_MODEL, test_file: str = "data/prontoqa_5hop_test.json"):
    """
    Loads the fine-tuned model and computes accuracy on the ProntoQA test set.
    """
    print(f"--- 📊 Evaluating Model from {model_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model and tokenizer
    tokenizer = get_llm_tokenizer()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    except OSError:
        print(f"Error: No model found at {model_path}.")
        return
        
    model.eval()

    # 2. Load ProntoQA test data
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        if isinstance(test_data, dict):
            test_data = list(test_data.values())
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file}.")
        return

    print(f"Loaded {len(test_data)} test samples.")
    correct = 0
    total = 0
    
    for sample in tqdm(test_data, desc="Evaluating"):
        prompt = sample.get('question', '')
        # True answer in ProntoQA is usually a boolean or string "True"/"False"
        true_answer = str(sample.get('answer', '')).lower()
        
        if not prompt:
            continue
            
        # 3. Generate response
        # Note: If your model expects <LATENT_i> tokens to start, 
        # ensure the prompt matches your Stage 2 training format.
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128, # ProntoQA reasoning is usually shorter than GSM8K
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Ensure it doesn't just stop immediately if using latents
                do_sample=False 
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 4. Check for correctness
        pred_answer = extract_prontoqa_answer(generated_text)
        
        # Clean the true answer for comparison (True/Yes -> true)
        if true_answer in ["true", "yes", "1"]:
            true_label = "true"
        else:
            true_label = "false"

        if pred_answer == true_label:
            correct += 1
        total += 1

    # 5. Report accuracy
    if total > 0:
        accuracy = (correct / total) * 100
        print("\n--- 📈 Evaluation Results ---")
        print(f"Correct: {correct}")
        print(f"Total:   {total}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("-------------------------------")
    else:
        print("No samples were evaluated.")

if __name__ == "__main__":
    # You can pass a different path for the OOD test set here
    evaluate_model(test_file="data/prontoqa_5hop_test.json")
