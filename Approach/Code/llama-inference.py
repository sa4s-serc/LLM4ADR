from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

MODEL_NAME = "google/flan-t5-base"
CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_PATH = "../models/flan-t5-base"
# MODEL_PATH = "/scratch/llm4adr/results/flan-t5-base/checkpoint-100"
EMBEDDING_MODEL = "bert-base-uncased"

pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

def perform_rag(query: str, top_k: int = 2) -> str:
    results = db.similarity_search(query, k=top_k + 1)
    
    for result in results:
        if result.page_content == query:
            results.remove(result)
            break
    
    if len(results) != top_k:
        results = results[:top_k]
    
    context = ''
    # context = "An architectural decision record is used to keep track of decisions made while building the project. It generally consists of a context and decision. You are an expert architect and are tasked with taking decisions given a particular context.\n Here are some examples:\n\n"
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        
    # context += f"Now provide a decision according to the context given below:\n{query}\n## Decision\n"
    context += query + "\n## Decision\n"

    return context

def infer(model, tokenizer, data, device) -> pd.DataFrame:
    inputs = data["Context"].apply(lambda x: perform_rag(x, 5))
    batch_size = 32
    
    predictions = []
        
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size].tolist()
        # print(batch[0], '-------------------')
        input_ids = tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(input_ids["input_ids"], max_length=1000)
        decodes = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(decodes[0])
        predictions.extend(decodes)
        # break
        
    print("Maximum Length of Prediction: ", max(len(p) for p in predictions))
    print("Maximum Length of Decision: ", max(len(d) for d in data["Decision"]))
    print("Average Length of Prediction: ", sum(len(p) for p in predictions) / len(predictions))
    print("Average Length of Decision: ", sum(len(d) for d in data["Decision"]) / len(data["Decision"]))
        
    data["Predictions"] = predictions
    
    return data

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR).to(device)

    device = None
    tokenizer = None
    model = None
    
    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True)
    data["Context"] = data["Context"].str.replace("\\n", "\n")
            
    results = infer(model, tokenizer, data, device)
    results.to_json(f"../results/{MODEL_NAME.split('/')[1]}-all.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()