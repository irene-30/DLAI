#!/bin/bash

# 1. Create data directory if it doesn't exist
mkdir -p data

# 2. Clone the ProntoQA source (The Data Factory)
if [ ! -d "prontoqa_source" ]; then
    echo "--- Cloning ProntoQA source repository ---"
    git clone https://github.com/asaparov/prontoqa.git prontoqa_source
fi

# 3. Generate the 5-hop reasoning dataset
echo "--- Generating 11,000 logical reasoning samples (5-hops) ---"
cd prontoqa_source
python3 run_experiment.py \
    --model-name json \
    --num-trials 11000 \
    --max-hops 5 \
    --output_file ../data/prontoqa_5hop.json

# 4. Return to project root and partition the data
cd ..
echo "--- Partitioning data into Train and Test splits ---"
python3 scripts/download_and_partition_prontoqa.py

echo "--- ✅ Dataset Setup Complete ---"
echo "Files created in data/: prontoqa_5hop_train.json, prontoqa_5hop_test.json"
