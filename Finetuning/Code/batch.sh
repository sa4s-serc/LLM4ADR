#!/bin/bash
#SBATCH -n 40
#SBATCH -w gnode069
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=12:00:00

source /home2/adyansh/miniconda3/bin/activate
conda activate /home2/adyansh/LLM4ADR/research

cd /home2/adyansh/LLM4ADR/Finetuning/Code

python3 local-script.py
