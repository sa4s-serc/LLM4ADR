from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

MODEL_NAME = "google/flan-t5-base"
CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_PATH = "../models/flan-t5-base"
MODEL_PATH = "/scratch/llm4adr/cache/Flan-T5-ADR/checkpoint-460"
EMBEDDING_MODEL = "bert-base-uncased"

pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

def count_tokens(text: str, tokenizer: AutoTokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)

def perform_rag(query: str, qid: int, tokenizer: AutoTokenizer, top_k: int = 2) -> str:
    retrieved = db.similarity_search(query, k=15)

    results = []
    
    for result in retrieved:
        if result.metadata['id'] != qid and count_tokens(result.page_content + result.metadata["Decision"], tokenizer) <= 1000:
            results.append(result)
            
        if len(results) == top_k:
            break
    
    if len(results) != top_k:
        raise Exception("not enough retrieved")
    
    context = ''
    # context = "An architectural decision record is used to keep track of decisions made while building the project. It generally consists of a context and decision. You are an expert architect and are tasked with taking decisions given a particular context.\n Here are some examples:\n\n"
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        
    # context += f"Now provide a decision according to the context given below:\n{query}\n## Decision\n"
    context += query + "\n## Decision\n"

    return context

def infer(model, tokenizer, data, device) -> pd.DataFrame:
    inputs = []

    for i, row in data.iterrows():
        inputs.append(perform_rag(row["Context"], qid=row["id"], tokenizer=tokenizer, top_k = 2))
        
    batch_size = 32
    
    predictions = []
        
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size]
        print(batch[0], '-------------------')
        input_ids = tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(input_ids["input_ids"], max_length=1000)
        decodes = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(decodes[0])
        predictions.extend(decodes)
        # break
        
    print("Maximum Length of Prediction: ", max(len(p) for p in predictions))
    print("Maximum Length of Decision: ", max(len(d) for d in data["Decision"]))
    print("Average Length of Prediction: ", sum(len(p) for p in predictions) / len(predictions))
    print("Average Length of Decision: ", sum(len(d) for d in data["Decision"]) / len(data["Decision"]))
        
    data["Predictions"] = predictions
    
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR).to(device)

    # device = None
    # tokenizer = None
    # model = None
    
    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True)
    data["Context"] = data["Context"].str.replace("\\n", "\n")
    
    results = infer(model, tokenizer, data, device)
    results.to_json(f"../results/{MODEL_NAME.split('/')[1]}-{MODEL_PATH.split('-')[-1]}.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()
