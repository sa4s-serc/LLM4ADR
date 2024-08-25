from copy import deepcopy
from typing import NamedTuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
import pandas as pd
import torch
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers.integrations import WandbCallback
import os
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import wandb
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(raise_error_if_not_found=True))

os.environ["WANDB_LOG_MODEL"]="false"
os.environ["WANDB_WATCH"]="false"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBEDDING_MODEL = "bert-base-uncased"
huggingface_token: str | None = os.getenv("HUGGINGFACE_TOKEN")
wandb.init(project="adr_novel_llama")
CACHE_DIR = "/scratch/llm4adr/cache"

torch_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, padding_side='left', token=huggingface_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-test.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

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

new_model = "llama-3-8b-ADR-approach"

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

train = pd.read_json(TRAIN_PATH, lines=True)
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

def perform_rag(query: str, top_k: int = 2) -> str:
    # Get one more result than required to remove the query from the results
    results = db.similarity_search(query, k=top_k + 1)
    
    # Remove the query from the results
    for result in results:
        if result.page_content == query:
            results.remove(result)
            break
    
    if len(results) != top_k:
        results = results[:top_k]
    
    few_shot = ""
    for result in results:
        few_shot += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"

    return few_shot

train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)

def format_chat_template(row):
    few_shot = perform_rag(row["Context"], 5)
    row_json = [
        {"role": "system", "content": "You are an expert architect and are tasked with taking decisions given a particular context. Here are some examples:\n\n" + few_shot},
        {"role": "user", "content": f"Provide a decision given the context below:\n{row['Context']}"},
        {"role": "assistant", "content": row["Decision"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

train_dataset = train_dataset.map(
    format_chat_template,
    # num_proc=4,
)

val_dataset = val_dataset.map(
    format_chat_template,
    # num_proc=4,
)

test_dataset = test_dataset.map(
    format_chat_template,
    # num_proc=4,
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
    eval_accumulation_steps=ACCUMULATION_STEPS,
    optim="paged_adamw_32bit",
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=1,
    # save_steps=1,
    logging_strategy="epoch",
    # logging_steps=1,
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
)
trainer.add_callback(CustomCallback(trainer))

trainer.train()

# tested = trainer.predict(test_dataset)


