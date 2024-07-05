from evaluate import load
import pandas as pd
import nltk
from BARTScore.bart_score import BARTScorer
import json

CACHE_DIR = '/scratch/adyansh/cache' 
# Any directory for temporary storage
DATA_DIR = '../results/result1.csv'
PREDICTION_COL = 'Predictions'
TRUE_COL = 'Output'
RESULT_DIR = '../results/score1.json'

def calculate_scores(cache_dir: str, data: pd.DataFrame, pred_col: str, true_col: str, result_dir: str) -> None:
    nltk.data.path.append(CACHE_DIR)
    rouge = load('rouge', cache_dir=CACHE_DIR)
    bleu = load('bleu', cache_dir=CACHE_DIR)
    meteor = load('meteor', cache_dir=CACHE_DIR)
    bertscore = load("bertscore", cache_dir=CACHE_DIR)
    bartscore = BARTScorer(device='cuda:1', cache_dir=CACHE_DIR)
    data.rename(columns={'Prediction': 'babbage-002'}, inplace=True)
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
    results['bartscore_recall'] = bartscore.score(data[PREDICTION_COL].tolist(), data[TRUE_COL].tolist(), batch_size = 4)
    results['bartscore_precision'] = bartscore.score(data[TRUE_COL].tolist(), data[PREDICTION_COL].tolist(), batch_size = 4)
    cols = ['precision', 'recall', 'f1']
    for c in cols:
        results['bertscore'][c] = pd.Series(results['bertscore'][c]).mean()
    results['bartscore_recall'] = pd.Series(results['bartscore_recall']).mean()
    results['bartscore_precision'] = pd.Series(results['bartscore_precision']).mean()
    results['bartscore_f1'] = (results['bartscore_recall'] + results['bartscore_precision'])/2
    with open(result_dir, 'w') as f:json.dump(results, f)
def main():
    calculate_scores(CACHE_DIR, pd.read_csv(DATA_DIR), PREDICTION_COL, TRUE_COL, RESULT_DIR)

if __name__ == '__main__':
    main()