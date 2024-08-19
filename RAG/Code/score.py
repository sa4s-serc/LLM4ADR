from evaluate import load
import pandas as pd
import nltk
import json
import sys

MODEL_NAME = sys.argv[1]
CACHE_DIR = '/scratch/llm4adr/cache' 
DATA_DIR = f'../results/{MODEL_NAME}.jsonl'
RESULT_DIR = f'../metrics/{MODEL_NAME}.json'
PREDICTION_COL = 'Predicted'
TRUE_COL = 'Decision'

def calculate_scores(data: pd.DataFrame) -> None:
    nltk.data.path.append(CACHE_DIR)
    rouge = load('rouge', cache_dir=CACHE_DIR)
    bleu = load('bleu', cache_dir=CACHE_DIR)
    meteor = load('meteor', cache_dir=CACHE_DIR)
    bertscore = load("bertscore", cache_dir=CACHE_DIR)
    
    # data.rename(columns={'Prediction': 'babbage-002'}, inplace=True)
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
    
    with open(RESULT_DIR, 'w') as f:
        json.dump(results, f)
    
def main():
    calculate_scores(pd.read_json(DATA_DIR, lines=True))

if __name__ == '__main__':
    main()