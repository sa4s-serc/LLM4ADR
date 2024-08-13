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

load_dotenv(dotenv_path="../../.env")

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/scratch/llm4adr/cache"
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

os.environ["WANDB_PROJECT"]="adr_llama"
wandb.init(project="adr_llama")

torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=True,
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR, device_map="auto", torch_dtype='auto')
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, token=huggingface_token, cache_dir=CACHE_DIR, device_map="auto", attn_implementation=attn_implementation)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

new_model = "llama-3-8b-ADR"

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

train = pd.read_json(TRAIN_PATH, lines=True).sample(100)
# val = train.copy()
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Context"]},
               {"role": "assistant", "content": row["Decision"]}]
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

EPOCHS = 10
BATCH_SIZE = 1 # Effective batch size is BATCH_SIZE * gradient_accumulation_steps
SAVE_TOTAL_LIM = 4

training_arguments = TrainingArguments(
    output_dir=f'{CACHE_DIR}/{new_model}',
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    logging_strategy="epoch",
    logging_steps=1,
    learning_rate=2e-4,
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
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    callbacks=[WandbCallback()],
)

trainer.train()