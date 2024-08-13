from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import torch
import sys
from transformers.integrations import WandbCallback
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import wandb

os.environ["WANDB_LOG_MODEL"]="false"
os.environ["WANDB_WATCH"]="false"

MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "bert-base-uncased"
# MODEL_NAME = "gpt2"
# MODEL_NAME = sys.argv[1]
# if MODEL_NAME is None:
#     print("Please provide a model name")
#     exit(1)

CACHE_DIR = "/scratch/llm4adr/cache"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
pkl = open(f'../../RAG/embeds/{EMBEDDING_MODEL}.pkl', 'rb').read()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_DIR)
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)

if MODEL_NAME == "gpt2":
    model  = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    os.environ["WANDB_PROJECT"]="adr_novel_gpt2"
    wandb.init(project="adr_novel_gpt2")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    os.environ["WANDB_PROJECT"]="adr_novel_t5"
    wandb.init(project="adr_novel_t5")

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

train = pd.read_json(TRAIN_PATH, lines=True)
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)

max_decision = int(max(train["Decision"].map(len).max(), val["Decision"].map(len).max(), test["Decision"].map(len).max()))

print("Max Decision Length: ", max_decision)

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
        
    context = ''
    # context = "An architectural decision record is used to keep track of decisions made while building the project. It generally consists of a context and decision. You are an expert architect and are tasked with taking decisions given a particular context.\n Here are some examples:\n\n"
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        
    # context += f"Now provide a decision according to the context given below:\n{query}\n## Decision\n"
    context += query + "\n## Decision\n"

    return context

def preprocess_function(data):
    inputs = data["Context"].apply(lambda x: perform_rag(x, 5))
    targets = data["Decision"]
    model_inputs = tokenizer(inputs.tolist(), padding="max_length", truncation=True)

    labels = tokenizer(text_target=targets.tolist(), padding="max_length", truncation=True)

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

train_dataset = FineTuningDataset(tokenized_train, tokenized_train["labels"])
val_dataset = FineTuningDataset(tokenized_val, tokenized_val["labels"])
test_dataset = FineTuningDataset(tokenized_test, tokenized_test["labels"])

BATCH_SIZE = 1 # Effective batch size is num_gpus * BATCH_SIZE ie 4
SAVE_TOTAL_LIM = 4
NUM_EPOCHS = 20

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
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
    generation_max_length=1000,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    run_name=None,
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
