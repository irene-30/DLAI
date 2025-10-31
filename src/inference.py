"""
Runs inference on a single custom question using the fine-tuned LLM.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM
from src.utils import get_llm_tokenizer, PATH_LLM_MODEL

def run_inference(question: str, model_path: str = PATH_LLM_MODEL):
    """
    Generates an answer for a single question.
    """
    print("--- 💬 Running Inference ---")
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
    
    # 2. Format prompt
    prompt = f"Question: {question}\nAnswer: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 3. Generate
    print(f"Prompt: {prompt}")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True, # Use sampling for more varied answers
            top_k=50,
            top_p=0.95
        )
        
    # 4. Decode and print
    # We set skip_special_tokens=False to see the latent tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    print("\n--- Generated Response (with latent tokens) ---")
    print(generated_text)
    
    # Optional: You can also print the "clean" version
    # clean_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print("\n--- Generated Response (Clean) ---")
    # print(clean_text)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a GSM8K question.")
    parser.add_argument(
        "-q", "--question", 
        type=str, 
        required=True, 
        help="The math question to ask."
    )
    parser.add_argument(
        "-m", "--model_path", 
        type=str, 
        default=PATH_LLM_MODEL,
        help="Path to the fine-tuned LLM (Stage 2)."
    )
    args = parser.parse_args()
    
    run_inference(args.question, args.model_path)