#!/bin/bash
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --output=output1.txt
#SBATCH --time=48:00:00
#SBATCH -w gnode084

source /home2/adyansh/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
mkdir -p /scratch/adyansh/cache
conda activate /home2/adyansh/LLM4ADR/research

python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/gemma.py --start 1000
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-base
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-small
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py t5-small
