#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=2-00:00:00
#SBATCH -w gnode062

source /home2/adyansh/miniconda3/bin/activate
conda activate /home2/adyansh/LLM4ADR/research

cd /home2/adyansh/LLM4ADR/Approach/Code

python3 inference.py
python3 score.py
