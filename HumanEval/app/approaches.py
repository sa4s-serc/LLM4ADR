import streamlit as st
from datetime import datetime
from streamlit import session_state as state
# from lorem import paragraph

import requests
import json

import random
import asyncio
from logger import logger

import google.generativeai as genai
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer

genai.configure(api_key=st.secrets['GEMINI'])
openai_client = openai.Client(api_key=st.secrets['OPENAI'])
HUGGINGFACE_TOKEN = st.secrets['HF_TOKEN']
CACHE_DIR = '/tmp'

pkl = open('text-embedding-3-large_eval.pkl', 'rb').read()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=st.secrets['OPENAI'])
vect_db_openai = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

pkl = open('bert-base-uncased_eval.pkl', 'rb').read()
embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased", cache_folder=CACHE_DIR)
vect_db_bert = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

system_message = "This is an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User."
error_message = "[ERROR]: An error occurred while generating decision. Please rate anything for this decision."

tokenizer_flant5 = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir=CACHE_DIR, max_length=1000, padding_side='left')
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=CACHE_DIR, model_max_length=4000, padding_side='left', token=HUGGINGFACE_TOKEN)
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", model_max_length=3000, padding_side='left', token=HUGGINGFACE_TOKEN)


@st.cache_data(ttl=0, persist=False)
def generating_decision(context):
    appr1 = random.choice([1, 2, 3])

    resp1 = ""
    if appr1 == 1:
        resp1 = asyncio.run(approach_one(context))
    elif appr1 == 2:
        resp1 = asyncio.run(approach_two(context))
    elif appr1 == 3:
        resp1 = asyncio.run(approach_three(context))

    if state.uid % 2 == 0:
        resp2 = asyncio.run(approach_four(context))
        appr2 = 4
    else:
        resp2 = asyncio.run(approach_five(context))
        appr2 = 5

    shuf = random.choice([0, 1])
    if shuf == 0:
        resp = [resp1, resp2]
        mapping = {1: f"Approach {appr1}", 2: f"Approach {appr2}"}
    else:
        resp = [resp2, resp1]
        mapping = {1: f"Approach {appr2}", 2: f"Approach {appr1}"}

    state.mapping = mapping
    for i, r in enumerate(resp):
        state.decisions[i].text = r


async def approach_one(context):
    "Zero shot inference from Gemini-1.5-pro"
    # return "Approach 1"

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        chat = model.start_chat(
            history=[
                {"role": "user", "parts": system_message}
            ]
        )
        response = chat.send_message(context)
        pred = response.text
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger("appr1", context=context, decision=pred, gen_time=gen_time)
        return pred

    except Exception as e:
        logger("error", error=f"Appr1: Unknown: {e}")
        return error_message


async def approach_two(context):
    "Retrieved few shot from GPT-4o"
    # return "Approach 2"

    try:
        def construct_context(query: str, db: FAISS, top_k: int = 5) -> str:
            try:
                retrieved = db.similarity_search(query, k=top_k+5)
                results = []
                for result in retrieved:
                    if (state.selected_context_id is not None and state.selected_context_id != result.metadata['id']) or (state.selected_context_id is None and query not in result.page_content):
                        results.append(result)

                if len(results) > top_k:
                    results = results[:top_k]

                if len(results) != top_k:
                    raise Exception("Not enough results found")

            except Exception as e:
                logger("error", error=f"Appr2: Error occurred: {e}")
                return error_message, []

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
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger('appr2', context=context, decision=pred, matched_ids=matched_ids, gen_time=gen_time)
        return pred

    except Exception as e:
        logger("error", error=f"Appr2: Unknown: {e}")
        return error_message


async def approach_three(context):
    "Finetuned gemma"
    # return "Approach 3"

    try:
        headers = {
            'Authorization': f'Bearer {st.secrets["HF_TOKEN"]}',
            'Content-Type': 'application/json',
        }

        messages = [{"role": "user", "content": context}]
        input_message = tokenizer_gemma.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        data = {"inputs": input_message}
        data = json.dumps(data)

        # logger("error", error=f"Appr3: Data: {data}")

        response = requests.post(
            st.secrets['HF_MODEL_FINE_TUNED_GEMMA'],
            headers=headers,
            data=data,
            timeout=60
        )
        if response.status_code != 200:
            logger("error", error=f"Appr3: Response failed: {response.status_code}, Response: {response.text}")
            return error_message
        response = response.json()[0]['generated_text']

        # logger("error", error=f"Appr3: Response: {response}")

        response = response[len(input_message):]
        # logger("error", error=f"Cleaned response - {response}")

        end_of_turn = tokenizer_gemma.convert_tokens_to_ids('<end_of_turn>')
        # logger("error", error=f"Appr3: EOT: {end_of_turn}")
        eot_position = response.find(tokenizer_gemma.decode([end_of_turn]))
        # logger("error", error=f"Appr3: EOT Position: {eot_position}")

        if eot_position != -1:
            response = response[:eot_position]

        # logger("error", error=f"Appr3: Response: {response}")

        response = response.replace(input_message, "").strip()
        response = response.replace("\\n", " \n ")
        response = "## Decision\n" + response
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger('appr3', context=context, decision=response, gen_time=gen_time)
        return response

    except Exception as e:
        logger("error", error=f"Appr3: Unknown: {e}")
        return error_message


def count_tokens(text: str, tokenizer: AutoTokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)


async def approach_four(context):
    "Novel approach using FlanT5"
    # return "Approach four"

    try:
        def perform_rag(query: str, db: FAISS, top_k: int = 2) -> str:
            try:
                retrieved = db.similarity_search(query, k=15)

                results = []
                retrieved_ids = []
                
                for result in retrieved:
                    if ((state.selected_context_id is not None and state.selected_context_id != result.metadata['id']) or (state.selected_context_id is None and query not in result.page_content)) and (count_tokens(result.page_content + result.metadata["Decision"], tokenizer_flant5) <= 1000):
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

            except Exception as e:
                logger("error", error=f"Appr4: Retrieval: {e}")
                return error_message, []

        updated_context, ids = perform_rag(context, vect_db_bert)
        if len(ids) == 0:
            return error_message

        headers = {
            'Authorization': f'Bearer {st.secrets["HF_TOKEN"]}',
            'Content-Type': 'application/json',
        }

        data = {"inputs": updated_context}
        data = json.dumps(data)
        # logger("error", error=f"Appr4: Data: {data}")
        # logger("error", error=f"Number of tokens: {count_tokens(updated_context, tokenizer_flant5)}")

        response = requests.post(
            st.secrets['HF_MODEL_NOVEL_ONE'],
            headers=headers,
            data=data,
            timeout=180
        )
        if response.status_code != 200:
            logger("error", error=f"Appr4: Response failed: {response.status_code}, Response: {response.text}")
            return error_message

        response = response.json()[0]['generated_text']
        # logger("error", error=f"Appr4: Response: {response}")
        response = response.replace("<pad>", "").strip()
        response = response.replace('\\n', ' \n ')
        response = response.replace('.n', ' \n ')
        response = "## Decision\n" + response
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger('appr4', context=context, decision=response, gen_time=gen_time, matched_ids=ids)
        return response

    except Exception as e:
        logger("error", error=f"Appr4: Unknown: {e}")
        return error_message


async def approach_five(context):
    "Novel approach using LLAMA"
    # return "Approach five"

    try:
        def perform_rag(query: str, db: FAISS, top_k: int = 2) -> str:
            try:
                retrieved = db.similarity_search(query, k=15)

                retrieved_ids = []
                results = []

                for result in retrieved:
                    if ((state.selected_context_id is not None and state.selected_context_id != result.metadata['id']) or (state.selected_context_id is None and query not in result.page_content)) and (count_tokens(result.page_content + result.metadata["Decision"], tokenizer_llama) <= 1000):
                        results.append(result)

                    if len(results) == top_k:
                        break

                if len(results) != top_k:
                    raise Exception("Not enough results found")

                few_shot = ""
                for result in results:
                    few_shot += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
                    retrieved_ids.append(result.metadata['id'])

            except Exception as e:
                logger("error", error=f"Appr5: Retrieval: {e}")
                return error_message, []

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
        # logger("error", error=f"Appr5: Input Message: {input_message}")

        data = {
            "inputs": input_message,
            "parameters": {"max_new_tokens": 500}
        }
        data = json.dumps(data)

        headers = {
            'Authorization': f'Bearer {st.secrets["HF_TOKEN"]}',
            'Content-Type': 'application/json',
        }

        response = requests.post(
            st.secrets['HF_MODEL_NOVEL_TWO'],
            headers=headers,
            data=data,
            timeout=180
        )
        if response.status_code != 200:
            logger("error", error=f"Appr5: Response failed: {response.status_code}, Response: {response.text}")
            return error_message

        response = response.json()[0]['generated_text']
        # logger("error", error=f"Appr5: Response: {response}")
        response = response.replace(input_message, "").strip()
        response = response.replace('\\n', ' \n ')
        response = "## Decision\n" + response
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger('appr5', context=context, decision=response, gen_time=gen_time, matched_ids=ids)
        return response

    except Exception as e:
        logger("error", error=f"Appr5: Unknown: {e}")
        return error_message
