"""
Utility script to download the GSM8K dataset from Hugging Face
and save it to the local `data/raw` directory as JSONL files.

This makes the project self-contained and reproducible without
relying on the HF hub after the first download.
"""

import os
from datasets import load_dataset

# Define the output directory (relative to the project root)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'gsm8k_train.jsonl')
TEST_FILE = os.path.join(OUTPUT_DIR, 'gsm8k_test.jsonl')

def download_and_save():
    print(f"Downloading gsm8k dataset...")
    try:
        dataset = load_dataset("gsm8k", "main")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and Hugging Face access.")
        return

    # Create the directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the 'train' split
    try:
        print(f"Saving 'train' split to {TRAIN_FILE}...")
        train_data = dataset['train']
        train_data.to_json(TRAIN_FILE, orient='records', lines=True)
        print(f"Successfully saved {len(train_data)} training samples.")
    except Exception as e:
        print(f"Error saving 'train' split: {e}")

    # Save the 'test' split
    try:
        print(f"Saving 'test' split to {TEST_FILE}...")
        test_data = dataset['test']
        test_data.to_json(TEST_FILE, orient='records', lines=True)
        print(f"Successfully saved {len(test_data)} test samples.")
    except Exception as e:
        print(f"Error saving 'test' split: {e}")

    print("\nData download complete.")

if __name__ == "__main__":
    download_and_save()
