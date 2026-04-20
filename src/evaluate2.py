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
    #parse_sample,
    extract_final_answer,
    PATH_LLM_MODEL
)

from peft import PeftModel
from transformers import AutoTokenizer

def evaluate_model(model_path):
    # 1. Load the original base model name (e.g., "meta-llama/Llama-3-8B")
    # This must match LLM_MODEL_NAME from your training
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct" 
    
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
    # 2. Load test data
    test_data = load_dataset("gsm8k", "main")['test']
    print(f"Loaded {len(test_data)} test samples.")
    correct = 0
    total = 0
    
    for sample in tqdm(test_data, desc="Evaluating"):
        parsed = parse_sample(sample)
        if not parsed:
            print(f"Skipping sample {i}: Failed to parse.")
            continue
        
        #prompt, _, solution = parsed
        prompt = parsed['prompt']
        solution = parsed['solution']
        
        # Get ground truth answer
        true_answer = extract_final_answer(solution)
        if true_answer is None:
            print(f"Skipping sample {i}: Failed to extract true answer.")
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
        print(total)

    # 5. Report accuracy
    print(total)
    accuracy = (correct / total) * 100
    print("\n--- 📈 Evaluation Results ---")
    print(f"Correct: {correct}")
    print(f"Total:   {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-------------------------------")

if __name__ == "__main__":
    evaluate_model()
