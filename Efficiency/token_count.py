from transformers import AutoTokenizer
import pandas as pd

from dotenv import load_dotenv
import os
load_dotenv()

HUGGINGFACE_TOKEN = os.environ['HF_TOKEN']
CACHE_DIR = '/tmp'

tokenizer_flant5 = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir=CACHE_DIR, max_length=1000, padding_side='left')
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=CACHE_DIR, model_max_length=4000, padding_side='left', token=HUGGINGFACE_TOKEN)
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", model_max_length=3000, padding_side='left', token=HUGGINGFACE_TOKEN)


def count_tokens(text: str, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)


mapping = {
    '3': ['input', 'response', tokenizer_gemma],
    '4': ['input', 'response', tokenizer_flant5],
    '5': ['input', 'output', tokenizer_llama],
}

def get_seconds(time_str):
    hours, minutes, seconds = time_str.split(":")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total_seconds


def analyse(approach: str):
    appr_data = pd.read_json(f'approach_{approach}.json', lines=True)

    token_data = pd.DataFrame(columns=['input_tokens', 'output_tokens', 'time'])
    token_data['input_tokens'] = appr_data[mapping[approach][0]].apply(lambda x: count_tokens(x, mapping[approach][2])) if approach not in ['1', '2'] else appr_data['input_tokens']
    token_data['output_tokens'] = appr_data[mapping[approach][1]].apply(lambda x: count_tokens(x, mapping[approach][2])) if approach not in ['1', '2'] else appr_data['output_tokens']
    token_data['time'] = appr_data['time'].apply(lambda x: get_seconds(x))

    token_data.to_json(f'approach_{approach}_tokens.json', orient='records', lines=True)


def analayse_flant5_gpu():
    appr_data = pd.read_json('approach_4_gpu.json')
    print(appr_data.columns)
    print(appr_data.head())

    token_data = pd.DataFrame(columns=['input_tokens', 'output_tokens', 'time'])
    token_data['input_tokens'] = appr_data['input'].apply(lambda x: count_tokens(x, tokenizer_flant5))
    token_data['output_tokens'] = appr_data['response'].apply(lambda x: count_tokens(x, tokenizer_flant5))
    token_data['time'] = appr_data['time'].apply(lambda x: get_seconds(x))
    print(token_data.head())

    token_data.to_json('approach_4_gpu_tokens.json', orient='records', lines=True)

# analyse('1')
# analyse('2')
# analyse('3')
# analyse('4')
# analyse('5')

analayse_flant5_gpu()

