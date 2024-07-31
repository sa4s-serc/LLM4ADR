import pandas as pd

DATA_TEST = "../../Data/ADR-data/data_test.jsonl"
MODEL_NAME = "gpt-3.5-turbo"
RESULT_DIR = f"../../0_shot/results/{MODEL_NAME}.jsonl"

test_data = pd.read_json(DATA_TEST, lines=True)["id"].tolist()
results = pd.read_json(RESULT_DIR, lines=True)

test_results = results[results["id"] == test_data[0]]
for i in range(1, len(test_data)):
    test_results = pd.concat([test_results, results[results["id"] == test_data[i]]])


test_results.to_json(f"{MODEL_NAME}.jsonl", orient="records", lines=True)
