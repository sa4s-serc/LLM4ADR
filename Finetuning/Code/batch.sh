#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=4-00:00:00
#SBATCH -w gnode059

source /home2/adyansh/miniconda3/bin/activate
conda activate /home2/adyansh/LLM4ADR/research

cd /home2/adyansh/LLM4ADR/Finetuning/Code

# python3 training.py google/flan-t5-base
# python3 training.py gpt2
# python3 lora-inference.py models/gemma-without-system
# python3 lora-inference.py models/gemma-2-9b
python3 score.py gemma-without-system
python3 score.py gemma-2-9b
# python3 lora-training.py
