from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser

load_dotenv()

INPUT_FILE = "../../Data/ADR-data/data.jsonl"

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE")
SCRATCH_DIR = "/scratch/llm4adr/cache"
MODEL_NAME = "gemma2"

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it", token=HUGGINGFACE_TOKEN, cache_dir=SCRATCH_DIR
)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HUGGINGFACE_TOKEN,
    cache_dir=SCRATCH_DIR,
)

system_message = "This is an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User."

def save_jsonl(data: pd.DataFrame, file_path, append=True):
    if append:
        with open(file_path, "a") as file:
            file.write(data.to_json(orient="records", lines=True))
    else:
        data.to_json(file_path, orient="records", lines=True)


def get_data(data: pd.DataFrame, max_length=-1):
    data["Context"] = system_message + data["Context"]

    try:
        done_files = [
            f for f in os.listdir("../results") 
            if f.startswith(f"{MODEL_NAME}_") and f.endswith(".jsonl") and not f.startswith(f"{MODEL_NAME}_failed")
        ]

        done = pd.DataFrame(columns=data.columns)
        for file in done_files:
            done = pd.concat([done, pd.read_json(f"../results/{file}", lines=True)], ignore_index=True)
        
        done_ids = done["id"].tolist()
        data = data[~data["id"].isin(done_ids)]
        print(f"Already done for {len(done_ids)} records")
    except Exception as e:
        print(f"Error occurred: {e}")

    if max_length != -1:
        removed = []
        data_new = pd.DataFrame(columns=data.columns)
        for i, row in data.iterrows():
            if len(row["Context"]) < max_length and len(row["Decision"]) < max_length:
                data_new = data_new.append(row, ignore_index=True)
            else:
                removed.append(row["id"])
        return data_new, removed
    return data, []


def run(start=0, num_left=10000):
    data = pd.read_json(INPUT_FILE, lines=True)
    print(f"Generating predictions using {MODEL_NAME}")

    data, removed = get_data(data)
    done = 0

    total = len(data[start:start+num_left]) if num_left > len(data[start:start+num_left]) else num_left
    print(f"Total records: {total}")

    failed = []

    for i, row in tqdm(data[start : start + num_left].iterrows(), total=total):
        if i in removed:
            continue

        try:
            input_ids = tokenizer(row["Context"], return_tensors="pt").to("cuda")
            outputs = model.generate(**input_ids, max_new_tokens=1000)
            row["Prediction"] = tokenizer.decode(outputs[0])
            row["GenTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = row.to_frame().T
            save_jsonl(row, f"../results/{MODEL_NAME}_{start}_{num_left}.jsonl")
            done += 1

        except Exception as e:
            print(f"Exception: {e}")
            print(f"Failed for {row['id']}")
            failed.append((row["id"], e))

        if done % 100 == 0:
            print(f"Done for {done} records")

    print(f"Done for {done} records")

    if len(failed) > 0:
        failed_df = data[data["id"].isin([f[0] for f in failed])]
        failed_df["Error"] = [f[1] for f in failed]
        save_jsonl(
            failed_df, f"../results/{MODEL_NAME}_failed_{start}_{num_left}.jsonl", False
        )
        print(f"Failed for {len(failed)} records")


if __name__ == "__main__":
    parser = ArgumentParser(prog="Gemma")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--num_left", type=int, default=10000, help="Number of runs left")
    args = parser.parse_args()

    run(args.start, args.num_left)
