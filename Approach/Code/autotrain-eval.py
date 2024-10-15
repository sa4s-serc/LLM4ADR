from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset
import sys
from transformers.integrations import WandbCallback

CACHE_DIR = "/scratch/llm4adr/cache"

# base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
base_model = "google/gemma-2-9b-it"
adapter_model = sys.argv[1]
# adapter_model = "rudradhar/test-100-gemma"

if "llama" in adapter_model:
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

huggingface_token_RD_personal = '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'
huggingface_token_RD_research = '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'
huggingface_token = huggingface_token_RD_research

tokenizer = AutoTokenizer.from_pretrained(base_model, token=huggingface_token, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model, token=huggingface_token, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto', max_length=3096)
model = PeftModel.from_pretrained(model, adapter_model, token=huggingface_token, cache_dir=CACHE_DIR)
model.eval()

# TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../retrieved/val.jsonl"
# TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

# train = pd.read_json(TRAIN_PATH, lines=True).sample(10, random_state=42)
val = pd.read_json(VAL_PATH, lines=True)
# test = pd.read_json(TEST_PATH, lines=True).sample(0)

# inputs = [v for v in val["text"]]
# labels = [v for v in val["Decision"]]

val_dataset = Dataset.from_pandas(val)

# def tokenize_function(examples):
#     tokenized_output = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=3096)
#     # tokenized_output["labels"] = tokenized_output["input_ids"].copy()  # Shift tokens for labels
#     return tokenized_output

def tokenize_function(examples):
    full_text: str = examples['text']

    user_input_start = full_text.find("<start_of_turn>user")
    model_output_start = full_text.find("<start_of_turn>model")
    
    user_input = full_text[user_input_start:model_output_start].strip()
    model_output = full_text[model_output_start:].strip()
#     print(user_input, model_output, '-'*100)

    concatenated_text = user_input + model_output

    tokenized_input = tokenizer(concatenated_text, truncation=True, padding="max_length", max_length=3000)

    input_length = len(tokenizer(user_input, truncation=True, max_length=3000)['input_ids'])
    
    tokenized_input['labels'] = [-100] * input_length + tokenized_input['input_ids'][input_length:]

    return tokenized_input

tokenized_dataset = val_dataset.map(tokenize_function)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=1,
    do_train=False,
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    callbacks=[WandbCallback()],
)

eval_results = trainer.evaluate()

# write the results to a json file in results folder
with open(f"./results/{adapter_model.split('/')[-1]}_eval_results.json", "w") as f:
    f.write(str(eval_results))
