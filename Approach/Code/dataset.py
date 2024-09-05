import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from datasets import Dataset

load_dotenv(find_dotenv(raise_error_if_not_found=True))

MODEL_NAME = "google/gemma-2-9b-it"
EMBEDDING_MODEL = "bert-base-uncased"
huggingface_token: str | None = os.getenv("HUGGINGFACE_TOKEN")
# wandb.init(project="adr_novel_gemma")
CACHE_DIR = "/scratch/llm4adr/cache"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, padding_side='right', token=huggingface_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

def count_tokens(text: str) -> int:
    encoding = tokenizer(text, return_tensors="pt")
    return encoding.input_ids.size(1)

def perform_rag(query: str, qid: int, top_k: int = 5) -> str:
    results = db.similarity_search(query, k=top_k+20, labels=True)
    
    # Filter out the exact match
    results = [result for result in results if result.metadata['id'] != qid and count_tokens(result.page_content + result.metadata["Decision"]) < 1000]

    # Ensure we only have top_k results
    if len(results) > top_k:
        results = results[:top_k]
    
    if len(results) != top_k:
        raise Exception("Not enough results found")
        
    few_shot = ""
    for result in results:
        few_shot += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"

    return few_shot

def format_chat_template(row):
    few_shot = perform_rag(row["Context"], row["id"], 2)
    row_json = [
        {"role": "user", "content": f"You are an expert architect and are tasked with taking decisions given a particular context. Here are some examples:\n\n{few_shot}\nProvide a decision given the context below:\n{row['Context']}"},
        {"role": "model", "content": row["Decision"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["text"] = row["text"].replace("\\n", "\n")
    return row

train = pd.read_json(TRAIN_PATH, lines=True)
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

# train = train.apply(format_chat_template, axis=1)
# val = val.apply(format_chat_template, axis=1)
test = test.apply(format_chat_template, axis=1)

# train.to_json("../retrieved/train.jsonl", lines=True, orient="records")
# val.to_json("../retrieved/val.jsonl", lines=True, orient="records")
test.to_json("../retrieved/test.jsonl", lines=True, orient="records")
