#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH -w gnode080
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=12:00:00

source /home2/adyansh/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
conda activate /home2/adyansh/LLM4ADR/research

cd /home2/adyansh/LLM4ADR/RAG/Code

python3 inference.py meta-llama/Meta-Llama-3-8B-Instruct
python3 inference.py google/gemma-2-9b-it
python3 score.py meta-llama/Meta-Llama-3-8B-Instruct-5
python3 score.py google/gemma-2-9b-it-5