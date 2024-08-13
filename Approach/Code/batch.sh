#!/bin/bash
#SBATCH -n 28
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=12:00:00
#SBATCH -w gnode074

source /home2/adyansh/miniconda3/bin/activate
conda activate /home2/adyansh/LLM4ADR/research

cd /home2/adyansh/LLM4ADR/Approach/Code

python3 training.py
