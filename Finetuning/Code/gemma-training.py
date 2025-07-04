from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from transformers.integrations import WandbCallback
import pandas as pd
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from dotenv import load_dotenv
import os
import wandb
import numpy as np

load_dotenv(dotenv_path="../../.env")

# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "google/gemma-2-9b-it"
CACHE_DIR = "/scratch/llm4adr/cache"
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# os.environ["WANDB_PROJECT"]="adr_llama"
# wandb.init(project="adr_llama")
os.environ["WANDB_PROJECT"]="adr_gemma"
wandb.init(project="adr_gemma")

torch_dtype = torch.float16
attn_implementation = 'eager'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print(f"Using {tokenizer.padding_side} padding with token {tokenizer.pad_token}")

# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto', attn_implementation=attn_implementation)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

# new_model = "llama-3-8b-ADR"
new_model = "gemma-2-9b-ADR-3"

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

train = pd.read_json(TRAIN_PATH, lines=True)
# val = train.copy()
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)

def format_chat_template(row):
    row_json = [{"role": "user", "content": f'You are an expert software architect. Your task is to help other software architects write Architectural Decision Records (ADRs). You will be given the context for the ADR and you need to provide the decision. The context is as follows: {row["Context"]}. Please provide the decision.'},
               {"role": "model", "content": row["Decision"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

train_dataset = train_dataset.map(
    format_chat_template,
    num_proc=4,
)

val_dataset = val_dataset.map(
    format_chat_template,
    num_proc=4,
)

test_dataset = test_dataset.map(
    format_chat_template,
    num_proc=4,
)

class CustomCallback(WandbCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # if control.should_evaluate:
        control_copy = deepcopy(control)
        self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train_epoch")
        return control_copy

EPOCHS = 10
BATCH_SIZE = 1 # Effective batch size is BATCH_SIZE * gradient_accumulation_steps
SAVE_TOTAL_LIM = 4
ACCUMULATION_STEPS = 8

training_arguments = TrainingArguments(
    output_dir=f'{CACHE_DIR}/{new_model}',
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    eval_accumulation_steps=2,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    logging_steps=1,
    group_by_length=True,
    report_to="wandb",
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    run_name=None,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    # callbacks=[WandbCallback()],
    # compute_metrics=compute_metrics
)
trainer.add_callback(CustomCallback(trainer))

trainer.train()
