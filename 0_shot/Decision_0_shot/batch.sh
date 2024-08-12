#!/bin/bash
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --output=output.txt
#SBATCH --time=72:00:00
#SBATCH -w gnode078

source /home2/ameyk/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
mkdir -p /scratch/llm4adr/cache
conda activate research

python3 /home2/ameyk/LLM4ADR/0_shot/Decision_0_shot/llama.py 
# python3 /home2/ameyk/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-base
# python3 /home2/ameyk/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-small
# python3 /home2/ameyk/LLM4ADR/0_shot/Decision_0_shot/score.py t5-small
