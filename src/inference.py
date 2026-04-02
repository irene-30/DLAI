"""
Runs inference on a single custom ProntoQA question using the fine-tuned Stage 2 LLM.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM
# Assuming these are updated in your src/utils.py to handle the new tokenizer
from src.utils import get_llm_tokenizer, PATH_LLM_MODEL

def run_inference(question: str, model_path: str = PATH_LLM_MODEL):
    """
    Generates an answer for a single logical reasoning question.
    """
    print(f"--- 💬 Running Inference on ProntoQA ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model and tokenizer
    # Ensure the tokenizer loaded here includes the <LATENT_0>... tokens
    tokenizer = get_llm_tokenizer()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    except OSError:
        print(f"Error: No model found at {model_path}.")
        print("Please run 'python src/train.py llm' first.")
        return
        
    model.eval()
    
    # 2. Format prompt 
    # For ProntoQA, we usually provide the context/question clearly.
    # The model is trained to start with Latents right after the 'Answer:' or 'Reasoning:'
    prompt = f"Question: {question}\nReasoning:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 3. Generate
    print(f"\n[Prompt]: {prompt}")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # For logic, top_p/sampling can be lower to keep reasoning strict
            do_sample=True, 
            top_k=40,
            top_p=0.9,
            temperature=0.7 
        )
        
    # 4. Decode
    # skip_special_tokens=False is essential to see your <LATENT_x> tokens!
    raw_generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    print("\n--- 🧠 Raw Response (Showing Latent Thoughts) ---")
    print(raw_generated_text)
    
    # Optional: Clean version for readability
    clean_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n--- ✨ Clean Response (Text Only) ---")
    print(clean_text)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a ProntoQA logic question.")
    parser.add_argument(
        "-q", "--question", 
        type=str, 
        required=True, 
        help="The logical reasoning question (e.g., 'Fae is a cat. Cats are mammals. Is Fae a mammal?')"
    )
    parser.add_argument(
        "-m", "--model_path", 
        type=str, 
        default=PATH_LLM_MODEL,
        help="Path to the fine-tuned LLM (Stage 2)."
    )
    args = parser.parse_args()
    
    run_inference(args.question, args.model_path)
