#!/bin/bash
#SBATCH -c 18
#SBATCH --gres=gpu:2
#SBATCH --output=output.txt
#SBATCH --time=48:00:00

source /home2/adyansh/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
mkdir -p /scratch/adyansh/cache
conda activate /home2/adyansh/LLM4ADR/research

python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/llama.py --num_left 1000 --start 2500
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-base
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-small
# python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py t5-small
