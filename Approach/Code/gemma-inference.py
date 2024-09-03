from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import os
from peft import PeftModel

load_dotenv(find_dotenv(raise_error_if_not_found=True))

MODEL_NAME = "google/gemma-2-9b-it"
CACHE_DIR = "/scratch/llm4adr/cache"
# MODEL_PATH = "../models/flan-t5-base"
MODEL_PATH = "../models/gemma-2-9b"
EMBEDDING_MODEL = "bert-base-uncased"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    encoding = tokenizer(text, return_tensors="pt")
    return encoding.input_ids.size(1)

def perform_rag(query: str, qid: int, tokenizer: AutoTokenizer, top_k: int = 5) -> str:
    results = db.similarity_search(query, k=top_k+20, labels=True)
    print(results[0])
    
    # Filter out the exact match
    results = [result for result in results if result.metadata['id'] != qid and count_tokens(result.page_content + result.metadata["Decision"], tokenizer) < 1000]

    # Ensure we only have top_k results
    if len(results) > top_k:
        results = results[:top_k]
    
    if len(results) != top_k:
        raise Exception("Not enough results found")
    
    context = ''
    # context = "An architectural decision record is used to keep track of decisions made while building the project. It generally consists of a context and decision. You are an expert architect and are tasked with taking decisions given a particular context.\n Here are some examples:\n\n"
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        
    # context += f"Now provide a decision according to the context given below:\n{query}\n## Decision\n"
    context += query + "\n## Decision\n"

    return context

def preprocess_texts(data: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
    prompts = []
    
    for index, row in data.iterrows():
        # print(row)
        few_shot = perform_rag(row["Context"], row["id"], tokenizer, 5)
        messages = [{"role": "user", "content": f'You are an expert architect and are tasked with taking decisions given a particular context. Here are some examples:\n\n{few_shot}\nProvide a decision given the context below:\n{row["Context"]}'}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    data["Prompts"] = prompts

def infer(model, tokenizer, data, device) -> pd.DataFrame:
    inputs = data["Prompts"]
    batch_size = 1
    
    predictions = []
        
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size].tolist()
        
        input_ids = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model.generate(**input_ids, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
        decodes = tokenizer.batch_decode(outputs)
        
        outputs = []
        for i in range(len(decodes)):
            outputs.append(decodes[i][len(batch[i]):])
        
        predictions.extend(outputs)
        
    print("Maximum Length of Prediction: ", max(len(p) for p in predictions))
    print("Maximum Length of Decision: ", max(len(d) for d in data["Decision"]))
    print("Average Length of Prediction: ", sum(len(p) for p in predictions) / len(predictions))
    print("Average Length of Decision: ", sum(len(d) for d in data["Decision"]) / len(data["Decision"]))
        
    data["Predictions"] = predictions
    
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, model_max_length=512, padding_side='left', token=HUGGINGFACE_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')
    
    model = PeftModel.from_pretrained(model, MODEL_PATH, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR)
    
    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True).sample(1, random_state=44)
    data["Context"] = data["Context"].str.replace("\\n", "\n")
    
    preprocess_texts(data, tokenizer)
            
    results = infer(model, tokenizer, data, device)
    results.to_json(f"../results/{MODEL_NAME.split('/')[1]}.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()