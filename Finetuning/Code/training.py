from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import evaluate
import pandas as pd
import torch
import time
import wandb
from transformers.integrations import WandbCallback
import os


os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"

# MODEL_NAME = "google/flan-t5-base"
MODEL_NAME = "gpt2"
CACHE_DIR = "/scratch/adyansh/cache"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
if MODEL_NAME == "gpt2":
    model  = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    os.environ["WANDB_PROJECT"]="adr_gpt2"
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    os.environ["WANDB_PROJECT"]="adr_t5"

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

train = pd.read_json(TRAIN_PATH, lines=True).sample(frac=0.01)
val = pd.read_json(VAL_PATH, lines=True).sample(frac=0.01)
test = pd.read_json(TEST_PATH, lines=True).sample(frac=0.01)

max_decision = int(max(train["Decision"].map(len).max(), val["Decision"].map(len).max(), test["Decision"].map(len).max()))

print("Max Decision Length: ", max_decision)

def preprocess_function(data):
    inputs = data["Context"]
    targets = data["Decision"]
    model_inputs = tokenizer(inputs.tolist(), padding="max_length", truncation=True)

    labels = tokenizer(text_target=targets.tolist(), padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = preprocess_function(train)
tokenized_val = preprocess_function(val)
tokenized_test = preprocess_function(test)

nltk.download("punkt", quiet=True)
rouge = evaluate.load("rouge")
bleu = evaluate.load('bleu', cache_dir=CACHE_DIR)
meteor = evaluate.load('meteor', cache_dir=CACHE_DIR)
bertscore = evaluate.load("bertscore", cache_dir=CACHE_DIR)

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

train_dataset = FineTuningDataset(tokenized_train, tokenized_train["labels"])
val_dataset = FineTuningDataset(tokenized_val, tokenized_val["labels"])
test_dataset = FineTuningDataset(tokenized_test, tokenized_test["labels"])

BATCH_SIZE = 1 # Effective batch size is num_gpus * BATCH_SIZE ie 4
SAVE_TOTAL_LIM = 4
NUM_EPOCHS = 20

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f'/scratch/adyansh/results/{MODEL_NAME}',
    evaluation_strategy="steps",
    logging_dir=f'./logs/{time.time()}/',
    logging_steps=100,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=SAVE_TOTAL_LIM,
    num_train_epochs=NUM_EPOCHS,
    push_to_hub=False,
    generation_max_length=1000,
    report_to="wandb",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[WandbCallback()],
    # compute_metrics=compute_metrics
)

training_data = trainer.train() 

trainer.evaluate(test_dataset)
