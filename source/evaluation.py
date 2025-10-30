"""
Evaluates the fine-tuned LLM (Stage 2) on the GSM8K test set.
"""
import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from src.utils import (
    get_llm_tokenizer, 
    parse_gsm8k_sample, 
    extract_final_answer,
    PATH_LLM_MODEL
)

def evaluate_model(model_path: str = PATH_LLM_MODEL):
    """
    Loads the fine-tuned model and computes accuracy on the test set.
    """
    print(f"--- 📊 Evaluating Model from {model_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model and tokenizer
    tokenizer = get_llm_tokenizer()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    except OSError:
        print(f"Error: No model found at {model_path}.")
        print("Please run 'python src/train.py llm' first.")
        return
        
    model.eval()

    # 2. Load test data
    test_data = load_dataset("gsm8k", "main")['test']
    
    correct = 0
    total = 0
    
    for sample in tqdm(test_data, desc="Evaluating"):
        parsed = parse_gsm8k_sample(sample)
        if not parsed:
            continue
        
        prompt, _, solution = parsed
        
        # Get ground truth answer
        true_answer = extract_final_answer(solution)
        if true_answer is None:
            continue
            
        # 3. Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256, # Allow space for reasoning
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 4. Check for correctness
        pred_answer = extract_final_answer(generated_text)
        
        if pred_answer is not None and pred_answer == true_answer:
            correct += 1
        total += 1

    # 5. Report accuracy
    accuracy = (correct / total) * 100
    print("\n--- 📈 Evaluation Results ---")
    print(f"Correct: {correct}")
    print(f"Total:   {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-------------------------------")

if __name__ == "__main__":
    evaluate_model()