from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv(raise_error_if_not_found=True))

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_PATH = "./models/llama-3-8b-ADR"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def infer(model, tokenizer, data, device) -> pd.DataFrame:
    inputs = data["Prompts"]
    batch_size = 8
    
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

def preprocess_texts(data: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
    prompts = []
    
    for i in data["Context"]:
        messages = [{"role": "user", "content": i}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    data["Prompts"] = prompts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, model_max_length=512)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')
    
    model = PeftModel.from_pretrained(model, MODEL_PATH, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR)

    model.eval()

    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True)
    data["Context"] = data["Context"].str.replace("\\n", "\n")
    
    preprocess_texts(data, tokenizer)
    
    results = infer(model, tokenizer, data, device)
    results.to_json(f"../results/{MODEL_NAME.split('/')[1]}.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()