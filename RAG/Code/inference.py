from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
import sys
from dotenv import find_dotenv, load_dotenv
import os
import gc

load_dotenv(find_dotenv(raise_error_if_not_found=True))

CACHE_DIR = "/scratch/llm4adr/cache"
EMBEDDING_MODEL = "bert-base-uncased"
MODEL_NAME = "google/gemma-2-9b-it"
MODEL_MAX_LENGTH = 500
RAG_DOCUMENTS = 5
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

pkl = open(f'../embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR, model_kwargs={'device': 'cpu'})
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

def construct_context(query: str, db: FAISS, embeddings: HuggingFaceEmbeddings, top_k: int = 2) -> str:
    results = db.similarity_search(query, k=top_k+1)
    
    for result in results:
        if result.page_content == query:
            results.remove(result)
            break
    
    if len(results) != top_k:
        results = results[:top_k]
    
    context = "You are an expert software architect who is tasked with making decisions for Architectural Decision Records (ADRs). You will be given a context and you need to provide a decision. Here are some examples:\n\n"
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        
    context += f"Make sure to give decisions that are similar to the ones above.\nNow provide a decision according to the context given below:\n{query}\n## Decision\n"
    
    return context

def replace_newline(text: list):
    for i in range(len(text)):
        text[i] = text[i].replace('\n', '\\n')
    return text
    
data = pd.read_json('../../Data/ADR-data/data_test.jsonl', lines=True)
context = data['Context']
print("Running Retrieval", flush=True)
rag_context = context.apply(lambda x: construct_context(x, db, embeddings, RAG_DOCUMENTS)).tolist()
print("Retrieval done", flush=True)
context = context.tolist()
decision = data['Decision'].tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, model_max_length=MODEL_MAX_LENGTH, token=huggingface_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')

model.generation_config.pad_token_id = tokenizer.pad_token_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


predicted_decision = []

BATCH_SIZE = 1

inputs = tokenizer(rag_context, return_tensors="pt", padding=True, truncation=True, max_length=MODEL_MAX_LENGTH, return_attention_mask=True)

with torch.no_grad():
    for i in tqdm(range(0, len(context), BATCH_SIZE)):
        input_ids = inputs['input_ids'][i:i+BATCH_SIZE].to(device)
        attention_mask = inputs['attention_mask'][i:i+BATCH_SIZE].to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=MODEL_MAX_LENGTH, min_length= int(MODEL_MAX_LENGTH/8))
        predicted_decision.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
        gc.collect()
        torch.cuda.empty_cache()

print(f"Prediction done for {len(predicted_decision)} records")

model = MODEL_NAME.split('/')[1]

df = pd.DataFrame({'Context': replace_newline(context), 'Decision': replace_newline(decision), 'Predicted': replace_newline(predicted_decision)})

# df.to_csv(f'../results/{model}.csv', index=False)
df.to_json(f'../results/{model}-{RAG_DOCUMENTS}.jsonl', orient='records', lines=True)