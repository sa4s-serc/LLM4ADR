from IPython import embed
import peft
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import TrainingArguments
import pandas as pd
import torch
import sys
from transformers.integrations import WandbCallback
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from copy import deepcopy

os.environ["WANDB_LOG_MODEL"]="false"
os.environ["WANDB_WATCH"]="false"

MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "bert-base-uncased"

CACHE_DIR = "/scratch/llm4adr/cache"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}-train.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM ,
    target_modules=["q", "v"]
)

# if MODEL_NAME == "gpt2":
#     model  = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#     os.environ["WANDB_PROJECT"]="adr_novel_gpt2"
#     wandb.init(project="adr_novel_gpt2")
# else:
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
os.environ["WANDB_PROJECT"]="adr_novel_t5"
wandb.init(project="adr_novel_t5")
    
model = get_peft_model(model, peft_config)

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

train = pd.read_json(TRAIN_PATH, lines=True)
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

#max_decision = int(max(train["Decision"].map(len).max(), val["Decision"].map(len).max(), test["Decision"].map(len).max()))

#print("Max Decision Length: ", max_decision)

def count_tokens(text: str) -> int:
    tokenized_input = tokenizer.tokenize(text)
    return len(tokenized_input)

def perform_rag(query: str, qid: int, top_k: int = 2) -> str:
    results = db.similarity_search(query, k=top_k+20, labels=True)
    
    # Filter out the exact match
    results = [result for result in results if result.metadata['id'] != qid and count_tokens(result.page_content + result.metadata["Decision"]) < 1000]

    # Ensure we only have top_k results
    if len(results) > top_k:
        results = results[:top_k]
    
    if len(results) != top_k:
        raise Exception("Not enough results found")
        
    context = ''
    # context = "An architectural decision record is used to keep track of decisions made while building the project. It generally consists of a context and decision. You are an expert architect and are tasked with taking decisions given a particular context.\n Here are some examples:\n\n"
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        
    # context += f"Now provide a decision according to the context given below:\n{query}\n## Decision\n"
    context += query + "\n## Decision\n"

    return context

def preprocess_function(data):
    inputs = []
    targets = []
    
    for _, row in data.iterrows():
        inputs.append(perform_rag(row["Context"], row["id"], 2))
        targets.append(row["Decision"])
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=3000)

    labels = tokenizer(text_target=targets, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = preprocess_function(train)
tokenized_val = preprocess_function(val)
tokenized_test = preprocess_function(test)

class FineTuningDataset(torch.utils.data.Dataset):
     def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels

     def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      return item

     def __len__(self):
      return len(self.labels)
  
del embeddings
del db
with torch.no_grad():
    torch.cuda.empty_cache()

train_dataset = FineTuningDataset(tokenized_train, tokenized_train["labels"])
val_dataset = FineTuningDataset(tokenized_val, tokenized_val["labels"])
test_dataset = FineTuningDataset(tokenized_test, tokenized_test["labels"])

class CustomCallback(WandbCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # if control.should_evaluate:
        control_copy = deepcopy(control)
        self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train_epoch")
        return control_copy


BATCH_SIZE = 1 # Effective batch size is num_gpus * BATCH_SIZE ie 4
SAVE_TOTAL_LIM = 4
NUM_EPOCHS = 20

# Set up training arguments
training_args = TrainingArguments(
    output_dir=f'/scratch/llm4adr/results/{MODEL_NAME.split("/")[-1]}',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    save_steps=1,
    save_total_limit=SAVE_TOTAL_LIM,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    push_to_hub=False,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    run_name=None,
    torch_empty_cache_steps=1,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[WandbCallback()],
    max_seq_length=3000
    # compute_metrics=compute_metrics
)
trainer.add_callback(CustomCallback(trainer))

training_data = trainer.train() 

trainer.evaluate(test_dataset)
