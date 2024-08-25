from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import os

MODEL_NAME = "google/flan-t5-base"
CACHE_DIR = os.path.expanduser("~/Desktop/ADR/cache")
MODEL_PATH = CACHE_DIR + "/Flan-T5-ADR/checkpoint-2946"

def infer(model, tokenizer, data) -> pd.DataFrame:
    inputs = data["Context"]
    batch_size = 8
    
    predictions = []
    
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size].tolist()
        input_ids = tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt")
        # outputs = model.generate(input_ids["input_ids"], max_length=1000)
        outputs = model.generate(**input_ids, max_length=1000, num_return_sequences=1)
        decodes = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decodes)
        
    print("Maximum Length of Prediction: ", max(len(p) for p in predictions))
    print("Maximum Length of Decision: ", max(len(d) for d in data["Decision"]))
    print("Average Length of Prediction: ", sum(len(p) for p in predictions) / len(predictions))
    print("Average Length of Decision: ", sum(len(d) for d in data["Decision"]) / len(data["Decision"]))
        
    data["Predictions"] = predictions
    
    return data

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = PeftModel.from_pretrained(model, MODEL_PATH, cache_dir=CACHE_DIR)

    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True)
    
    results = infer(model, tokenizer, data)
    results.to_json(f"../results/lora-Flan-T5-base.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()