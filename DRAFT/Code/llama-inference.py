from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from peft import PeftModel
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv
import os
import sys

load_dotenv(find_dotenv(raise_error_if_not_found=True))

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_PATH = "rudradhar/autotrain-llama-5"
# MODEL_PATH = sys.argv[1]
# MODEL_PATH = "/scratch/llm4adr/results/flan-t5-base/checkpoint-100"
EMBEDDING_MODEL = "bert-base-uncased"

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)
# tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, tokenizer: AutoTokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)

def perform_rag(query: str, tokenizer: AutoTokenizer, qid:int = 0, top_k: int = 2) -> str:
    retrieved = db.similarity_search(query, k=15)

    results = []
    
    for result in retrieved:
        if result.page_content != query and result.metadata['id'] != qid and count_tokens(result.page_content + result.metadata["Decision"], tokenizer) <= 1000:
            results.append(result)
            
        if len(results) == top_k:
            break
            
    few_shot = ""
    for result in results:
        few_shot += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"

    messages = [{"role": "system", "content": "You are an expert architect and are tasked with taking decisions given a particular context. Here are some examples:\n\n" + few_shot},
                {"role": "user", "content": f"Provide a decision given the context below:\n{query}"}]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def infer(model, tokenizer, data, device) -> pd.DataFrame:
    inputs = data["Prompts"]
    # batch_size = 1
    
    predictions = []
    
    for input in tqdm(inputs):
        # print(input, '-------------------')
        input_ids = tokenizer(input, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model.generate(**input_ids, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
        decodes = tokenizer.decode(outputs[0])
        
        # outputs = []
        # for i in range(len(decodes)):
        #     outputs.append(decodes[i][len(input[i]):])
        
        # print(decodes)
        predictions.append(decodes[len(input):])
        
    print("Maximum Length of Prediction: ", max(len(p) for p in predictions))
    print("Maximum Length of Decision: ", max(len(d) for d in data["Decision"]))
    print("Average Length of Prediction: ", sum(len(p) for p in predictions) / len(predictions))
    print("Average Length of Decision: ", sum(len(d) for d in data["Decision"]) / len(data["Decision"]))
        
    data["Predictions"] = predictions
    
    return data

def preprocess_texts(data: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
    prompts = []

    for i, row in data.iterrows():
        prompts.append(perform_rag(row["Context"], tokenizer, qid=row["id"]))

    data["Prompts"] = prompts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, model_max_length=4000, padding_side='left', token=HUGGINGFACE_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')
    
    model = PeftModel.from_pretrained(model, MODEL_PATH, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR)
    model.eval()

    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True)
    # data = data[data['id'] == 2309]
    data["Context"] = data["Context"].str.replace("\\n", "\n")
    
    preprocess_texts(data, tokenizer)
    
    # Sorting in descending order of prompt length so any cuda out of memory comes early
    data = data.reindex(data.Prompts.str.len().sort_values(ascending=False).index).reset_index(drop=True)
    
    results = infer(model, tokenizer, data, device)
    results.to_json(f"../results/{MODEL_PATH.split('/')[1]}.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()
