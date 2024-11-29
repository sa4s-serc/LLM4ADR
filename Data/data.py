import pandas as pd

def main():
    data = pd.read_json('ADR-data/data_test.jsonl', lines=True).sample(100)
    data.to_csv('data_test_sample.csv', index=False)
    
if __name__ == '__main__':
    main()