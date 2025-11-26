"""
Defines the main LLM with QLoRA support for Colab compatibility.
Loads Llama 3.2 in 4-bit and adds LoRA adapters.
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

def get_llm_model(model_name: str, tokenizer_len: int):
    """
    Loads Llama 3.2 in 4-bit precision and attaches LoRA adapters.
    
    Args:
        model_name: Hugging Face model ID (e.g. 'meta-llama/Llama-3.2-3B-Instruct')
        tokenizer_len: New vocabulary size (to resize embeddings)
    """
    print(f"Loading Llama model: {model_name} with QLoRA...")
    
    # 1. Quantization Config (4-bit)
    # This shrinks the model footprint by ~4x, allowing it to fit in 16GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 3. Resize embeddings for new tokens
    # Note: Resizing embeddings in QLoRA requires training the embedding layer.
    current_vocab = model.config.vocab_size
    if current_vocab != tokenizer_len:
        print(f"Resizing token embeddings from {current_vocab} to {tokenizer_len}...")
        model.resize_token_embeddings(tokenizer_len)

    # 4. Apply LoRA (Low-Rank Adaptation)
    # This makes the model trainable on T4 GPU by freezing the main weights
    # and only training small adapter matrices.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16,            # Rank: Higher = more parameters, more expressivity
        lora_alpha=32,   # Scaling factor
        lora_dropout=0.05,
        # Important: We must target all linear layers for best performance
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # CRITICAL: We added new tokens (<latent_x>), so we MUST train the embeddings
        modules_to_save=["embed_tokens", "lm_head"] 
    )
    
    model = get_peft_model(model, peft_config)
    
    print("Model converted to PEFT/LoRA mode.")
    model.print_trainable_parameters()
    
    return model
