import pandas as pd
import tiktoken
from sklearn.model_selection import train_test_split

def filter(num_tokens: int, file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    encoding = tiktoken.get_encoding("cl100k_base")
    df["tokens"] = df["Context"].apply(lambda x: len(encoding.encode(x)))
    df = df[df["tokens"] <= num_tokens]
    df["id"] = df.index
    df.to_json(file_path.replace(".csv", ".jsonl"), orient="records", lines=True)
    return df

def split(df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.2) -> None:
    train, val_test = train_test_split(df, test_size=val_size + test_size, random_state=42)
    val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size), random_state=42)
    train.to_json("../ADR-data/data_train.jsonl", orient="records", lines=True)
    val.to_json("../ADR-data/data_val.jsonl", orient="records", lines=True)
    test.to_json("../ADR-data/data_test.jsonl", orient="records", lines=True)
    
def main():
    df = filter(500, "../ADR-data/context_decision.csv")
    split(df)
    
if __name__ == "__main__":
    main()