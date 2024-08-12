from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm import tqdm

CACHE_DIR = "/scratch/llm4adr/cache"
EMBEDDING_MODEL = "bert-base-uncased"
MODEL_NAME = "google/flan-t5-base"
MODEL_MAX_LENGTH = 1000
RAG_DOCUMENTS = 2

pkl = open(f'../embeds/{EMBEDDING_MODEL}.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

def construct_context(query: str, db: FAISS, embeddings: HuggingFaceEmbeddings, top_k: int = 2) -> str:
    results = db.similarity_search(query, k=top_k)
    
    context = "Use the provided contexts and decisions to make a more informed decision:\n\n"
    for result in results:
        context += "## Context\n" + result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        
    context += f"Make sure to give decisions that are similar to the ones above.\nNow provide a decision according to the context given below:\n{query}\n## Decision\n"
    
    return context

def replace_newline(text: list):
    for i in range(len(text)):
        text[i] = text[i].replace('\n', '\\n')
    return text
    
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, model_max_length=MODEL_MAX_LENGTH)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map='auto')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_json('../../Data/ADR-data/data_test.jsonl', lines=True)
context = data['Context']
rag_context = context.apply(lambda x: construct_context(x, db, embeddings, RAG_DOCUMENTS)).tolist()
context = context.tolist()
decision = data['Decision'].tolist()

predicted_decision = []

BATCH_SIZE = 16

inputs = tokenizer(rag_context, return_tensors="pt", padding=True, truncation=True, max_length=MODEL_MAX_LENGTH, return_attention_mask=True)

with torch.no_grad():
    for i in tqdm(range(0, len(context), BATCH_SIZE)):
        input_ids = inputs['input_ids'][i:i+BATCH_SIZE].to(device)
        attention_mask = inputs['attention_mask'][i:i+BATCH_SIZE].to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=MODEL_MAX_LENGTH, min_length= int(MODEL_MAX_LENGTH/8))
        predicted_decision.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print(f"Prediction done for {len(predicted_decision)} records")

model = MODEL_NAME.split('/')[1]

df = pd.DataFrame({'Context': replace_newline(context), 'Decision': replace_newline(decision), 'Predicted': replace_newline(predicted_decision)})

# df.to_csv(f'../results/{model}.csv', index=False)
df.to_json(f'../results/{model}-{RAG_DOCUMENTS}.jsonl', orient='records', lines=True)