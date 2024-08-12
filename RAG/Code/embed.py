from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

CACHE_DIR = "/scratch/llm4adr/cache"
MODEL_NAME = "bert-base-uncased"

def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["Decision"] = record["Decision"]

    return metadata

# Load the text
loader = JSONLoader(
    file_path="../../Data/ADR-data/data_train.jsonl",
    jq_schema=".",
    content_key="Context",
    text_content=False,
    json_lines=True,
    metadata_func=metadata_func,
)
documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# print(docs[0])

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME, cache_folder=CACHE_DIR
)
db = FAISS.from_documents(documents, embeddings)

pkl = db.serialize_to_bytes()

with open(f"../embeds/{MODEL_NAME}.pkl", "wb") as f:
    f.write(pkl)
