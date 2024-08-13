import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import torch
import numpy as np
import logging
import torch

CACHE_DIR = '/scratch/llm4adr/cache'
CUDA_DEVICE = 'cuda:0' # 'cuda:0' or 'cuda:1' or 'auto'

data = pd.read_csv('../ADR-data/context_decision.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(data: pd.DataFrame, max_length = -1):
    context = data['Context'].tolist()
    decision = data['Decision'].tolist()
    for i in range(len(context)):
        context[i] = f"This is an Architectural Decision Record. Provide a Decision for the Context given below.\n{context[i]}\n## Decision\n"
    if max_length != -1:
        removed = []
        context_new = []
        decision_new = []
        for i, (c, d) in enumerate(zip(context, decision)):
            if len(c) <= max_length:
                context_new.append(c)
                decision_new.append(d)
            else:
                removed.append(i)
        context = context_new
        decision = decision_new
        
    return context, decision, removed

def replace_newline(text: list):
    for i in range(len(text)):
        text[i] = text[i].replace('\n', '\\n')
    print(len(text))
    return text

model_name = "google-t5/t5-base"
# model_name = "google/flan-t5-base"
model_max_length = 2500

context, decision, removed = get_data(data, model_max_length)
print(f"Data loaded for {len(context)} records")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, model_max_length=model_max_length)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=CACHE_DIR, device_map=CUDA_DEVICE)

predicted_decision = []

BATCH_SIZE = 16

inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=model_max_length, return_attention_mask=True)

with torch.no_grad():
    for i in tqdm(range(0, len(context), BATCH_SIZE)):
        input_ids = inputs['input_ids'][i:i+BATCH_SIZE].to(device)
        attention_mask = inputs['attention_mask'][i:i+BATCH_SIZE].to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=model_max_length, min_length= int(model_max_length/8))
        predicted_decision.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print(f"Prediction done for {len(predicted_decision)} records")

for i in removed:
    predicted_decision.insert(i, "")
    
model = model_name.split('/')[1]

df = pd.DataFrame({'Context': replace_newline(data['Context']), 'Decision': replace_newline(data['Decision']), model: replace_newline(predicted_decision)})

df.to_csv(f'../results/{model}.csv', index=False)
