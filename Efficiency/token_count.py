import tiktoken
from transformers import AutoTokenizer
import pandas as pd
from datetime import datetime
import time
from datetime import timedelta

from dotenv import load_dotenv
import os
load_dotenv()

HUGGINGFACE_TOKEN = os.environ['HF_TOKEN']
CACHE_DIR = '/tmp'

tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
tokenizer_flant5 = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir=CACHE_DIR, max_length=1000, padding_side='left')
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=CACHE_DIR, model_max_length=4000, padding_side='left', token=HUGGINGFACE_TOKEN)
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", model_max_length=3000, padding_side='left', token=HUGGINGFACE_TOKEN)


def count_tokens(text: str, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)


mapping = {
    '1': ['context', 'response', tokenizer_tiktoken],
    '2': ['fewshot', 'response', tokenizer_tiktoken],
    '3': ['input', 'response', tokenizer_gemma],
    '4': ['input', 'response', tokenizer_flant5],
    '5': ['input', 'output', tokenizer_llama],
}


def analyse(approach: str):
    appr_data = pd.read_json(f'approach_{approach}.json', lines=True)

    token_data = pd.DataFrame(columns=['input_tokens', 'output_tokens'])
    token_data['input_tokens'] = appr_data[mapping[approach][0]].apply(lambda x: count_tokens(x, mapping[approach][2]))
    token_data['output_tokens'] = appr_data[mapping[approach][1]].apply(lambda x: count_tokens(x, mapping[approach][2]))

    token_data.to_json(f'approach_{approach}_tokens.json', orient='records', lines=True)


analyse('1')
analyse('2')
analyse('3')
analyse('4')
analyse('5')
