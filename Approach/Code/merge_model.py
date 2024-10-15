from transformers import AutoModelForCausalLM
from peft import PeftModel
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv(raise_error_if_not_found=True))

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/scratch/llm4adr/cache"
ADAPTER_MODEL= "rudradhar/autotrain-llama-5"

HUGGINGFACE_TOKEN: str | None = os.getenv('HUGGINGFACE_TOKEN')

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR)

# merge the models
model = model.merge_and_unload()

# push the model to hub
model.push_to_hub(ADAPTER_MODEL.split('/')[1] + '-merged', token=HUGGINGFACE_TOKEN)