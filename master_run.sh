#!/bin/bash
set -e
rm -f questions.txt predictions.txt contexts.txt predictions.json squad_ground_truth.json

# 1. Prepare dev set questions and ground truth
python3 prep_reference.py

# 2. Run RAG (make sure main.py generates contexts.txt)
bash run.sh questions.txt predictions.txt

# 3. Convert predictions to JSON format
python3 converter.py

# 4. Evaluate with Recall
python3 evaluate_with_recall.py squad_ground_truth.json predictions.json contexts.txt