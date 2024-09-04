from evaluate import load
import pandas as pd
import nltk
import json
from argparse import ArgumentParser

CACHE_DIR = '/scratch/llm4adr/cache' 
PREDICTION_COL = 'Predictions'
TRUE_COL = 'Decision'

def calculate_scores(data: pd.DataFrame) -> None:
    nltk.data.path.append(CACHE_DIR)
    rouge = load('rouge', cache_dir=CACHE_DIR)
    bleu = load('bleu', cache_dir=CACHE_DIR)
    meteor = load('meteor', cache_dir=CACHE_DIR)
    bertscore = load("bertscore", cache_dir=CACHE_DIR)
    
    data = data.dropna(subset=PREDICTION_COL)
    
    print('Your data is of length: ', len(data))
    
    results = {}
    results['rouge'] = rouge.compute(predictions=data[PREDICTION_COL],references= data[TRUE_COL])
    print('Rouge Done')
    results['bleu'] = bleu.compute(predictions=data[PREDICTION_COL],references= data[TRUE_COL])
    print('Bleu Done')
    results['meteor'] = meteor.compute(predictions=data[PREDICTION_COL],references= data[TRUE_COL])
    print('Meteor Done')
    results['bertscore'] = bertscore.compute(predictions=data[PREDICTION_COL],references= data[TRUE_COL], lang='en', batch_size = 64)
    print('BertScore Done')
    cols = ['precision', 'recall', 'f1']
    for c in cols:
        results['bertscore'][c] = pd.Series(results['bertscore'][c]).mean()
    return results
    
def main(model_name: str) -> None:
    if not model_name:
        print("Please provide a model name")
        return
    print("Calculating score for:", model_name)
    data_dir = f'../results/{model_name}.jsonl'
    result_dir = f'../metrics/{model_name}.json'    

    results = calculate_scores(pd.read_json(data_dir, lines=True))

    with open(result_dir, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = ArgumentParser(prog="Score")
    parser.add_argument("--model", type=str, help="Name of the model")
    model_name = parser.parse_args().model

    main(model_name)