#!/bin/bash
#SBATCH -c 36
#SBATCH --gres=gpu:3
#SBATCH --output=output1.txt
#SBATCH --time=72:00:00
#SBATCH -w gnode074

source /home2/adyansh/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
mkdir -p /scratch/adyansh/cache
conda activate /home2/adyansh/LLM4ADR/research

python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/gemma.py
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-base
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-small
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py t5-small
