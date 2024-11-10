from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import dotenv
import os
import sys

dotenv.load_dotenv(dotenv.find_dotenv(raise_error_if_not_found=True))

MODEL_NAME = "google/gemma-2-9b-it"
CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_PATH = sys.argv[1]
# MODEL_PATH = "./models/gemma-2-9b"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def infer(model, tokenizer, data, device) -> pd.DataFrame:
    inputs = data["Prompts"]
    batch_size = 1
    
    predictions = []
    
    end_of_turn = tokenizer.convert_tokens_to_ids('<end_of_turn>')
    
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size].tolist()
        
        input_ids = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model.generate(**input_ids, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
        decodes = tokenizer.batch_decode(outputs)
        
        outputs = []
        for i in range(len(decodes)):
            outputs.append(decodes[i][len(batch[i]):])
            
            eot_position = outputs[i].find(tokenizer.decode([end_of_turn]))
            if eot_position != -1:
                outputs[i] = outputs[i][:eot_position]


        predictions.extend(outputs)
        
    print("Maximum Length of Prediction: ", max(len(p) for p in predictions))
    print("Maximum Length of Decision: ", max(len(d) for d in data["Decision"]))
    print("Average Length of Prediction: ", sum(len(p) for p in predictions) / len(predictions))
    print("Average Length of Decision: ", sum(len(d) for d in data["Decision"]) / len(data["Decision"]))
        
    data["Predictions"] = predictions
    
    return data

def preprocess_text(context: str, tokenizer: AutoTokenizer) -> None:    
    messages = [{"role": "user", "content": f'You are an expert software architect. Your task is to help other software architects write Architectural Decision Records (ADRs). You will be given the context for the ADR and you need to provide the decision. The context is as follows: {context}. Please provide the decision.'}]

    prompt = (tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    return prompt

def predict(model, tokenizer, input_str, device) -> str:
    end_of_turn = tokenizer.convert_tokens_to_ids('<end_of_turn>')
    
    inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    generated_text = decoded_output[len(input_str):]
    
    eot_position = generated_text.find(tokenizer.decode([end_of_turn]))
    if eot_position != -1:
        generated_text = generated_text[:eot_position]
    
    return generated_text


def preprocess_texts(data: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
    prompts = []
    
    for i in data["Context"]:
        if "system" in MODEL_PATH:
            messages = [{"role": "user", "content": i}]
        else:
            messages = [{"role": "user", "content": f'You are an expert software architect. Your task is to help other software architects write Architectural Decision Records (ADRs). You will be given the context for the ADR and you need to provide the decision. The context is as follows: {i}. Please provide the decision.'}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    data["Prompts"] = prompts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, model_max_length=512, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')
    
    model = PeftModel.from_pretrained(model, MODEL_PATH, token=HUGGINGFACE_TOKEN, cache_dir=CACHE_DIR)

    model.eval()

    data = pd.read_json("../../Data/ADR-data/data_test.jsonl", lines=True)
    data["Context"] = data["Context"].str.replace("\\n", "\n")
    

    preprocess_texts(data, tokenizer)
    
    results = infer(model, tokenizer, data, device)
    print(results)
    results.to_json(f"../results/{MODEL_PATH.split('/')[1]}.jsonl", lines=True, orient="records")
    
if __name__ == '__main__':
    main()
