from transformers import AutoModel;
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live;
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(raise_error_if_not_found=True))

model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir="/scratch/llm4adr/cache", token=os.getenv("HUGGINGFACE_TOKEN"))
print(estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1))