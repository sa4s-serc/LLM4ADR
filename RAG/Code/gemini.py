import google.generativeai as genai

from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


INPUT_FILE = "../../Data/ADR-data/data_test.jsonl"
CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_NAME = "gemini-1.5-pro"
RAG_DOCUMENTS = 5

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel(MODEL_NAME)

pkl = open(f'../embeds/gemini_test.pkl', 'rb').read()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_query")
db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)


system_message = "This is an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User."

def save_jsonl(data: pd.DataFrame, file_path, append=True):
    if append:
        with open(file_path, "a") as file:
            file.write(data.to_json(orient="records", lines=True))
    else:
        data.to_json(file_path, orient="records", lines=True)


def get_data(data: pd.DataFrame, max_length=-1):
    try:
        done_files = [
            f for f in os.listdir("../results")
            if f.startswith(f"{MODEL_NAME}_") and f.endswith(".jsonl") and not f.startswith(f"{MODEL_NAME}_failed")
        ]

        done = pd.DataFrame(columns=data.columns)
        for file in done_files:
            done = pd.concat([done, pd.read_json(f"../results/{file}", lines=True)], ignore_index=True)

        if not done.empty:
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


def construct_context(query: str, qid: int, db: FAISS, top_k: int = 5) -> str:
    results = db.similarity_search(query, k=top_k+1, labels=True)
    
    # Filter out the exact match
    results = [result for result in results if result.metadata['id'] != qid]

    # Ensure we only have top_k results
    if len(results) > top_k:
        results = results[:top_k]
    
    if len(results) != top_k:
        raise Exception("Not enough results found")
    
    context = "You are an expert software architect who is tasked with making decisions for Architectural Decision Records (ADRs). You will be given a context and you need to provide a decision. Here are some examples:\n\n"
    
    retrieved = []
    for result in results:
        context += result.page_content + "\n## Decision\n" + result.metadata['Decision'] + "\n\n"
        retrieved.append(result.metadata['id'])
    context += f"Make sure to give decisions that are similar to the ones above.\nNow provide a decision according to the context given below:\n{query}\n## Decision\n"
    
    return context, retrieved
   

def run(start=0):
    data = pd.read_json(INPUT_FILE, lines=True)
    print(f"Generating predictions using {MODEL_NAME}")

    data, removed = get_data(data)
    open(f"../results/{MODEL_NAME}_failed_{start}.jsonl", "w").close()
    done = 0

    total = len(data[start:])
    print(f"Total records: {total}")

    failed = []

    for i, row in tqdm(data[start:].iterrows(), total=total):
        if i in removed:
            continue

        try:
            system_message, ids = construct_context(row["Context"], row['id'], db, RAG_DOCUMENTS)
            chat = model.start_chat(
                history=[
                    {"role": "user", "parts": system_message}
                ]
            )
            response = chat.send_message(row["Context"])
            row["MatchedIds"] = ids
            row["Prediction"] = response.text
            row["GenTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = row.to_frame().T
            save_jsonl(row, f"../results/{MODEL_NAME}_{start}.jsonl")
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
        save_jsonl(failed_df, f"../results/{MODEL_NAME}_failed_{start}.jsonl", False)
        print(f"Failed for {len(failed)} records")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    run(args.start)
