import json
import openai
import os
from dotenv import load_dotenv
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime

INPUT_FILE = "../../Data/ADR-data/data.jsonl"

MODEL_NAME = "gpt-4o"
# MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = 'babbage-002'


def save_jsonl(data: pd.DataFrame, file_path, append=True):
    if append:
        with open(file_path, "a") as file:
            file.write(data.to_json(orient="records", lines=True))
    else:
        data.to_json(file_path, orient="records", lines=True)


def get_data(data: pd.DataFrame, max_length=-1):
    data["Context"] = data["Context"] + "\n## Decision\n"

    try:
        done = pd.read_json(f"../results/{MODEL_NAME}_0_10000.jsonl", lines=True)
        done = done["id"].tolist()
        data = data[~data["id"].isin(done)]
    except:
        pass

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


load_dotenv()

system_message = "This is an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User."

openai.api_key = os.getenv("OPENAI_KEY")

CHAT_MODELS = ["gpt-4o", "gpt-3.5-turbo"]
COMPLETION_MODELS = ["babbage-002"]


def run(start=0, days=10000):
    data = pd.read_json(INPUT_FILE, lines=True)
    print(f"Generating predictions using {MODEL_NAME}")

    data, removed = get_data(data)
    done = 0

    total = len(data[start:]) if days > len(data[start:]) else days
    print(f"Total records: {len(data[start:])}")

    failed = []

    for i, row in tqdm(data[start : start + days].iterrows(), total=total):
        if i in removed:
            continue

        try:
            if MODEL_NAME in CHAT_MODELS:
                response = openai.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": row["Context"]},
                    ],
                    max_tokens=1000,
                )
                row["Predicted"] = response.choices[0].message.content

            elif MODEL_NAME in COMPLETION_MODELS:
                response = openai.completions.create(
                    model=MODEL_NAME,
                    prompt="This is an Architectural Decision Record for a software"
                    + row["Context"]
                    + "\n## Decision\n",
                    max_tokens=1000,
                )
                row["Predicted"] = response.choices[0].text

            else:
                raise KeyError("Invalid model")

            row["GenTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = row.to_frame().T
            save_jsonl(row, f"../results/{MODEL_NAME}_{start}_{days}.jsonl")
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
            failed_df, f"../results/{MODEL_NAME}_failed_{start}_{days}.jsonl", False
        )
        print(f"Failed for {len(failed)} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GPT")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--day", type=int, default=10000, help="Number of days to run")
    args = parser.parse_args()

    run(args.start, args.day)
