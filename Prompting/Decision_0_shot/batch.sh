#!/bin/bash
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --output=output.txt
#SBATCH --time=4-00:00:00
#SBATCH -w gnode062

source ~/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
mkdir -p /scratch/llm4adr/cache
conda activate ~/LLM4ADR/research/

cd ~/LLM4ADR/0_shot/Decision_0_shot

python3 inference.py
