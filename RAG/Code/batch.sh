#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH -w gnode056
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=12:00:00

source /home2/adyansh/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
conda activate /home2/adyansh/LLM4ADR/research

python3 /home2/adyansh/LLM4ADR/RAG/Code/inference.py
python3 /home2/adyansh/LLM4ADR/RAG/Code/score.py