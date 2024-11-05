from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, BitsAndBytesConfig
from transformers.integrations import WandbCallback
import pandas as pd
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import Dataset
from dotenv import load_dotenv
import os
import wandb
from copy import deepcopy

MODEL_NAME = "google/flan-t5-base"
# CACHE_DIR = os.path.expanduser("~/Desktop/ADR/cache")
CACHE_DIR = "/scratch/llm4adr/cache"

os.environ["WANDB_PROJECT"]="adr_Flan_T5"
wandb.init(project="adr_Flan_T5")

torch_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # task_type="CAUSAL_LM", # for llama, gemma
    task_type=TaskType.SEQ_2_SEQ_LM , # for Flan-T5
    # target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'] # for llama, gemma
    target_modules=["q", "v"] # for Flan-T5
)
model = get_peft_model(model, peft_config)

TRAIN_PATH = "../../Data/ADR-data/data_train.jsonl"
VAL_PATH = "../../Data/ADR-data/data_val.jsonl"
TEST_PATH = "../../Data/ADR-data/data_test.jsonl"

train = pd.read_json(TRAIN_PATH, lines=True)
# val = train.copy()
val = pd.read_json(VAL_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)


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
new_model = "/Flan-T5-ADR"

training_arguments = TrainingArguments(
    output_dir=f'{CACHE_DIR}/{new_model}',
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    eval_accumulation_steps=ACCUMULATION_STEPS,
    # optim="paged_adamw_32bit",
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=2,
    # save_steps=1,
    logging_strategy="epoch",
    # logging_steps=1,
    # learning_rate=2e-4,
    group_by_length=True,
    report_to="wandb",
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    run_name=None,
    # eval_on_start=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    # dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    # callbacks=[WandbCallback()],
)
trainer.add_callback(CustomCallback(trainer))

trainer.evaluate()
trainer.train()