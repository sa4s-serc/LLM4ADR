#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=4-00:00:00
#SBATCH -w gnode067

source ~/miniconda3/bin/activate
conda activate ~/LLM4ADR/research

cd ~/LLM4ADR/Verifier/

python3 inference.py ../Approach/results/autotrain-llama-1.jsonl openai
#python3 inference.py ../Approach/results/autotrain-llama-1.jsonl gemini

python3	inference.py ../Approach/results/autotrain-gemma-5.jsonl openai
#python3 inference.py ../Approach/results/autotrain-gemma-5.jsonl gemini

python3	inference.py ../Finetuning/results/Meta-Llama-3-8B-Instruct.jsonl openai
#python3 inference.py ../Finetuning/results/Meta-Llama-3-8B-Instruct.jsonl gemini

python3	inference.py ../Finetuning/results/gemma-without-system.jsonl openai
#python3 inference.py ../Finetuning/results/gemma-without-system.jsonl gemini
