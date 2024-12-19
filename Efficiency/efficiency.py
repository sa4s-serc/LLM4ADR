import requests
import json

import pandas as pd
import asyncio

import google.generativeai as genai
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer

from datetime import datetime
import time
from datetime import timedelta


from argparse import ArgumentParser 

from dotenv import load_dotenv
import os
load_dotenv()

genai.configure(api_key=os.environ['GEMINI'])
openai_client = openai.Client(api_key=os.environ['OPENAI'])
HUGGINGFACE_TOKEN = os.environ['HF_TOKEN']
CACHE_DIR = '/tmp'

pkl = open('../HumanEval/app/text-embedding-3-large_eval.pkl', 'rb').read()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ['OPENAI'])
vect_db_openai = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

pkl = open('../HumanEval/app/bert-base-uncased_eval.pkl', 'rb').read()
embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased", cache_folder=CACHE_DIR)
vect_db_bert = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

system_message = "This is an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User."
error_message = "[ERROR]: An error occurred while generating decision. Please rate anything for this decision."

tokenizer_flant5 = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir=CACHE_DIR, max_length=1000, padding_side='left')
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=CACHE_DIR, model_max_length=4000, padding_side='left', token=HUGGINGFACE_TOKEN)
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", model_max_length=3000, padding_side='left', token=HUGGINGFACE_TOKEN)


async def approach_one(context, qid):
    "Zero shot inference from Gemini-1.5-pro"
    # return "Approach 1"
    log = {}

    time_start = datetime.now()
    model = genai.GenerativeModel("gemini-1.5-pro")
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": system_message}
        ]
    )
    response = chat.send_message(context)
    log['context'] = context
    log['response'] = response.text
    log['time'] = datetime.now() - time_start
    log['input_tokens'] = response.usage_metadata.prompt_token_count
    log['output_tokens'] = response.usage_metadata.candidates_token_count
    return log


async def approach_two(context, qid):
    "Retrieved few shot from GPT-4o"
    # return "Approach 2"
    log = {}

    time_start = datetime.now()
    def construct_context(query: str, db: FAISS, top_k: int = 5) -> str:
        retrieved = db.similarity_search(query, k=top_k+1)
        results = []
        for result in retrieved:
            if (qid != result.metadata['id']):
                results.append(result)

        if len(results) > top_k:
            results = results[:top_k]

        if len(results) != top_k:
            raise Exception("Not enough results found")

        context = "You are an expert software architect who is tasked with making decisions for Architectural Decision Records (ADRs). You will be given a context and you need to provide a decision. Here are some examples:\n\n"
        retrieved = []
        for result in results:
            context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
            retrieved.append(result.metadata['id'])
        context += "Make sure to give decisions that are similar to the ones above.\nNow provide a decision " \
            f"according to the context given below:\n{query}\n## Decision\n"

        return context, retrieved

    updated_context, ids = construct_context(context, vect_db_openai)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": updated_context},
            {"role": "user", "content": context},
        ],
        max_tokens=1000,
    )
    matched_ids = ids
    pred = response.choices[0].message.content

    log['context'] = context
    log['fewshot'] = updated_context
    log['response'] = pred
    log['matched_ids'] = matched_ids
    log['time'] = datetime.now() - time_start
    log['input_tokens'] = response.usage.prompt_tokens
    log['output_tokens'] = response.usage.completion_tokens

    return log


async def approach_three(context, qid):
    "Finetuned gemma"
    # return "Approach 3"
    log = {}
    time_start = datetime.now()

    headers = {
        'Authorization': f'Bearer {os.environ["HF_TOKEN"]}',
        'Content-Type': 'application/json',
    }

    messages = [{"role": "user", "content": context}]
    input_message = tokenizer_gemma.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    data = {"inputs": input_message}
    data = json.dumps(data)

    response = requests.post(
        os.environ['HF_MODEL_FINE_TUNED_GEMMA'],
        headers=headers,
        data=data,
        timeout=60
    )

    response = response.json()[0]['generated_text']
    response = response[len(input_message):]
    end_of_turn = tokenizer_gemma.convert_tokens_to_ids('<end_of_turn>')
    eot_position = response.find(tokenizer_gemma.decode([end_of_turn]))

    if eot_position != -1:
        response = response[:eot_position]

    log['context'] = context
    log['input'] = input_message
    log['response'] = response
    log['time'] = datetime.now() - time_start

    response = response.replace(input_message, "").strip()
    response = response.replace("\\n", " \n ")
    response = "## Decision\n" + response

    log['post'] = response

    return log


def count_tokens(text: str, tokenizer: AutoTokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)


async def approach_four(context, qid):
    "Novel approach using FlanT5"
    # return "Approach four"

    log = {}
    time_start = datetime.now()

    def perform_rag(query: str, db: FAISS, top_k: int = 2) -> str:
        retrieved = db.similarity_search(query, k=5)

        results = []
        retrieved_ids = []
        
        for result in retrieved:
            if (qid != result.metadata['id']) and (count_tokens(result.page_content + result.metadata["Decision"], tokenizer_flant5) <= 1000):
                results.append(result)

            if len(results) == top_k:
                break
        
        if len(results) != top_k:
            raise Exception("Not enough results found")
        
        context = ''
        for result in results:
            context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
            retrieved_ids.append(result.metadata['id'])

        context += query + "\n## Decision\n"
        return context, retrieved_ids

    updated_context, ids = perform_rag(context, vect_db_bert)
    if len(ids) == 0:
        return error_message

    headers = {
        'Authorization': f'Bearer {os.environ["HF_TOKEN"]}',
        'Content-Type': 'application/json',
    }

    data = {"inputs": updated_context}
    data = json.dumps(data)
    

    response = requests.post(
        os.environ['HF_MODEL_NOVEL_ONE'],
        headers=headers,
        data=data,
        timeout=180
    )
    if response.status_code != 200:
        return error_message
    response = response.json()[0]['generated_text']
    
    log['context'] = context
    log['input'] = updated_context
    log['response'] = response
    log['matched_ids'] = ids
    log['time'] = datetime.now() - time_start

    response = response.replace("<pad>", "").strip()
    response = response.replace('\\n', ' \n ')
    response = response.replace('.n', ' \n ')
    response = "## Decision\n" + response

    log['post'] = response

    return log


async def approach_five(context, qid):
    "Novel approach using LLAMA"
    # return "Approach five"

    log = {}
    time_start = datetime.now()

    def perform_rag(query: str, db: FAISS, top_k: int = 2) -> str:
        retrieved = db.similarity_search(query, k=5)

        retrieved_ids = []
        results = []

        for result in retrieved:
            if (qid != result.metadata['id']) and (count_tokens(result.page_content + result.metadata["Decision"], tokenizer_llama) <= 1000):
                results.append(result)

            if len(results) == top_k:
                break

        if len(results) != top_k:
            raise Exception("Not enough results found")

        few_shot = ""
        for result in results:
            few_shot += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
            retrieved_ids.append(result.metadata['id'])

        return few_shot, retrieved_ids

    few_shot_examples, ids = perform_rag(context, vect_db_bert)
    if len(ids) == 0:
        return error_message

    messages = [
        {
            "role": "system",
            "content": f"You are an expert architect and are tasked with taking decisions given a particular context. Here are some examples:\n\n{few_shot_examples}"
        },
        {
            "role": "user",
            "content": f"Provide a decision given the context below:\n{context}"
        }
    ]
    
    input_message = tokenizer_llama.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    data = {
        "inputs": input_message,
        "parameters": {"max_new_tokens": 500}
    }
    data = json.dumps(data)

    headers = {
        'Authorization': f'Bearer {os.environ["HF_TOKEN"]}',
        'Content-Type': 'application/json',
    }

    response = requests.post(
        os.environ['HF_MODEL_NOVEL_TWO'],
        headers=headers,
        data=data,
        timeout=180
    )
    if response.status_code != 200:
        return error_message
    response = response.json()[0]['generated_text']

    log['context'] = context
    log['input'] = input_message
    log['response'] = response
    log['matched_ids'] = ids
    log['time'] = datetime.now() - time_start

    response = response.replace(input_message, "").strip()
    response = response.replace('\\n', ' \n ')
    response = "## Decision\n" + response

    log['post'] = response

    return log

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, timedelta):
            return str(obj)  # Or obj.total_seconds()
        return super().default(obj)



def get_response(data, approach_num, sleep_time=10):
    func = None
    if approach_num == 1:
        func = approach_one
    elif approach_num == 2:
        func = approach_two
    elif approach_num == 3:
        func = approach_three
    elif approach_num == 4:
        func = approach_four
    elif approach_num == 5:
        func = approach_five

    for ind, row in data.iterrows():
        context = row['context']
        qid = row['id']
        response = asyncio.run(func(context, qid))
        with open(f'approach_{approach_num}.json', 'a') as f:
            f.write(json.dumps(response, cls=CustomJSONEncoder) + '\n')
        print(f"Aproach {approach_num} completed for {ind+1}/{len(data)}")
        time.sleep(sleep_time)

    # df = pd.DataFrame(results)
    # df.to_json(f'approach_{approach_num}.json', orient='records')
    print(f"Approach {approach_num} completed")


data = pd.read_csv('efficiency.csv')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--a", type=int, default=1, help="Approach number to run")
    parser.add_argument("--s", type=int, default=10, help="Sleep time between each request")
    args = parser.parse_args()

    get_response(data, args.a, args.s)
