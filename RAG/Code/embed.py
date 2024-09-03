from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv(raise_error_if_not_found=True))

print(os.environ.get("GOOGLE_API_KEY"))
print(os.environ.get("OPENAI_API_KEY"))

CACHE_DIR = "/scratch/llm4adr/cache"
# MODEL_NAME = "bert-base-uncased" | "text-embedding-3-large" | "models/text-embedding-004"
MODEL_NAME = "bert-base-uncased"

def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["Decision"] = record["Decision"]
    metadata["id"] = record["id"]

    return metadata

def main():
    print()
    loader = JSONLoader(
        file_path="../../Data/ADR-data/data_test.jsonl",
        jq_schema=".",
        content_key="Context",
        text_content=False,
        json_lines=True,
        metadata_func=metadata_func,
    )
    documents = loader.load()
    # documents = documents[:10]

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    # print(docs[0])

    if "text-embedding-3" in MODEL_NAME:
        embeddings = OpenAIEmbeddings(model=MODEL_NAME)
    elif "models" in MODEL_NAME:
        embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_NAME)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME, cache_folder=CACHE_DIR
        )
    db = FAISS.from_documents(documents, embeddings)

    pkl = db.serialize_to_bytes()

    with open(f"../embeds/{MODEL_NAME}-test.pkl", "wb") as f:
        f.write(pkl)

if __name__ == "__main__":
    main()