"""
Utility script to partition the generated ProntoQA dataset into 
Train and Test splits for the Token Assorted replication.

This ensures the project is self-contained and follows the 
5-hop reasoning structure defined in the experiment setup.
"""

import os
import json
import random

# Define the local paths (relative to the project root)
# PATH_RAW_DATA should point to the output of:
# python run_experiment.py --model-name json --max-hops 5 --output_file ...
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'prontoqa_5hop.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'prontoqa_5hop_train.json')
TEST_FILE = os.path.join(OUTPUT_DIR, 'prontoqa_5hop_test.json')

def partition_data(test_split: float = 0.1):
    """
    Loads the generated ProntoQA JSON and partitions it into train/test sets.
    """
    print(f"--- 📂 Partitioning ProntoQA Dataset ---")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        print("Please run the ProntoQA generation command first:")
        print("python run_experiment.py --model-name json --num-trials 11000 --max-hops 5 --output_file data/prontoqa_5hop.json")
        return

    # 1. Load the generated data
    try:
        with open(RAW_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        # ProntoQA generation often outputs a dictionary; convert to a list of samples
        if isinstance(data, dict):
            samples = list(data.values())
        else:
            samples = data
            
        print(f"Successfully loaded {len(samples)} total samples.")
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # 2. Shuffle and Split
    random.seed(42)  # For reproducibility
    random.shuffle(samples)
    
    num_test = int(len(samples) * test_split)
    test_samples = samples[:num_test]
    train_samples = samples[num_test:]

    # 3. Save Splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        print(f"Saving {len(train_samples)} training samples to {TRAIN_FILE}...")
        with open(TRAIN_FILE, 'w') as f:
            json.dump(train_samples, f, indent=2)
            
        print(f"Saving {len(test_samples)} test samples to {TEST_FILE}...")
        with open(TEST_FILE, 'w') as f:
            json.dump(test_samples, f, indent=2)
            
    except Exception as e:
        print(f"Error saving splits: {e}")

    print("\n--- ✅ Data partitioning complete ---")

if __name__ == "__main__":
    partition_data()
