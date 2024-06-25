#!/bin/bash
#SBATCH -n 10
#SBATCH -w gnode080
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=12:00:00

source /home2/adyansh/miniconda3/bin/activate

# eval "$(conda shell.bash hook)"
conda activate /home2/adyansh/LLM4ADR/research

python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py t5-base
python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-base
python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py flan-t5-small
python3 /home2/adyansh/LLM4ADR/0_shot/Decision_0_shot/score.py t5-small