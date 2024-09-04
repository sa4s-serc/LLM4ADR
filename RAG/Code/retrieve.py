from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from datasets import Dataset
import pandas as pd

huggingface_token: str | None = os.getenv("HUGGINGFACE_TOKEN")

CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_NAME = "google/gemma-2-9b-it"
EMBEDDING_MODEL = "bert-base-uncased"

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"
RAG_DOCUMENTS = 2

train = pd.read_json(TRAIN_PATH, lines=True)
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, padding_side='left', token=huggingface_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)


def perform_rag(query: str, qid: int, db: FAISS, top_k: int = RAG_DOCUMENTS) -> str:
    results = db.similarity_search(query, k=top_k+1, labels=True)
    
    # Filter out the exact match
    results = [result for result in results if result.metadata['id'] != qid]

    # Ensure we only have top_k results
    if len(results) > top_k:
        results = results[:top_k]
    
    if len(results) != top_k:
        raise Exception("Not enough results found")
    
    context = "You are an expert software architect who is tasked with making decisions for Architectural Decision Records (ADRs). You will be given a context and you need to provide a decision. Here are some examples:\n\n"
    
    retrieved = []
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        retrieved.append(result.metadata['id'])
    context += f"Make sure to give decisions that are similar to the ones above.\nNow provide a decision according to the context given below:\n{query}\n## Decision\n"
    
    return context, retrieved
    


train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)

def format_chat_template(row):
    few_shot, ret_list = perform_rag(row["Context"], RAG_DOCUMENTS)
    row_json = [
        {"role": "user", "content": f"You are an expert architect and are tasked with taking decisions given a particular context. Here are some examples:\n\n{few_shot}\nProvide a decision given the context below:\n{row['Context']}"},
        {"role": "model", "content": row["Decision"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

train_dataset = train_dataset.map(
    format_chat_template,
    # num_proc=4,
)

val_dataset = val_dataset.map(
    format_chat_template,
    # num_proc=4,
)

test_dataset = test_dataset.map(
    format_chat_template,
    # num_proc=4,
)

train_dataset.save_to_disk("../../Data/ADR-data/train_dataset")
val_dataset.save_to_disk("../../Data/ADR-data/val_dataset")
test_dataset.save_to_disk("../../Data/ADR-data/test_dataset")

