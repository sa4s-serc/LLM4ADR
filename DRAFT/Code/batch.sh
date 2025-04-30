#!/bin/bash
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output3.txt
#SBATCH --time=4-00:00:00
#SBATCH -w gnode062

source ~/miniconda3/bin/activate
conda activate ~/LLM4ADR/research

cd ~/LLM4ADR/Approach/Code

# python3 inference.py
# python3 llama-training.py
# python3 gemma-training.py


# python3 autotrain-eval.py rudradhar/autotrain-gemma-3-v2
# python3 autotrain-eval.py rudradhar/autotrain-gemma-4-v2


#python3 gemma-inference.py rudradhar/autotrain-gemma-1-v2
#python3 score.py autotrain-gemma-1-v2
#python3 gemma-inference.py rudradhar/autotrain-gemma-2-v2
#python3 score.py autotrain-gemma-2-v2
#python3 gemma-inference.py rudradhar/autotrain-gemma-3-v2
#python3 score.py autotrain-gemma-3-v2
#python3 gemma-inference.py rudradhar/autotrain-gemma-4-v2
#python3 score.py autotrain-gemma-4-v2

#python3 gemma-inference.py rudradhar/autotrain-gemma-3
#python3 score.py autotrain-gemma-3
#python3 gemma-inference.py rudradhar/autotrain-gemma-1
#python3 score.py autotrain-gemma-1
#python3 llama-inference.py rudradhar/autotrain-llama-1
#python3 score.py autotrain-llama-1

python3 autotrain-eval.py rudradhar/autotrain-llama-2
