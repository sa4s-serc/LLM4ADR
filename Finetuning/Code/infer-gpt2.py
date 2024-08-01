from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import torch
from tqdm import tqdm

MODEL_NAME = "gpt2"
CACHE_DIR = "/scratch/adyansh/cache"
MODEL_PATH = "./models/gpt2"

def infer(tokenizer, data, device) -> pd.DataFrame:
    inputs = data["Context"]
    batch_size = 32
    
    predictions = []
    
    generator = pipeline('text-generation', model=MODEL_PATH, tokenizer=MODEL_NAME, device=device)
    
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size].tolist()
        output = generator(batch, max_length=1000, pad_token_id=tokenizer.eos_token_id, truncation=True)
        outputs = [o[0]["generated_text"] for o in output]
        # print(output)
        predictions.extend(outputs)
        
    print("Maximum Length of Prediction: ", max(len(p) for p in predictions))
    print("Maximum Length of Decision: ", max(len(d) for d in data["Decision"]))
    print("Average Length of Prediction: ", sum(len(p) for p in predictions) / len(predictions))
    print("Average Length of Decision: ", sum(len(d) for d in data["Decision"]) / len(data["Decision"]))
        
    data["Predictions"] = predictions
    
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True)
    
    results = infer(tokenizer, data, device)
    results.to_json(f"../results/{MODEL_NAME}.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()