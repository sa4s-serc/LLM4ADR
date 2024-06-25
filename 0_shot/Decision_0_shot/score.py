from evaluate import load
import pandas as pd
import nltk
from BARTScore.bart_score import BARTScorer
import sys

CACHE_DIR = '/scratch/adyansh/cache'
MODEL_NAME = sys.argv[1]
nltk.data.path.append(CACHE_DIR)

rouge = load('rouge', cache_dir=CACHE_DIR)
bleu = load('bleu', cache_dir=CACHE_DIR)
meteor = load('meteor', cache_dir=CACHE_DIR)
bertscore = load("bertscore", cache_dir=CACHE_DIR)
bartscore = BARTScorer(cache_dir=CACHE_DIR, device='cuda:0')

print("Loaded models", flush=True)

data = pd.read_csv(f'../results/{MODEL_NAME}.csv')

print(len(data), flush=True)
data.head()

# remove rows in df where MODEL_NAME column is empty
data = data.dropna(subset=[MODEL_NAME])
len(data)

results = {}

results['rouge'] = rouge.compute(predictions=data[MODEL_NAME],references= data['Decision'])
print('Rouge Done', flush=True)
results['bleu'] = bleu.compute(predictions=data[MODEL_NAME],references= data['Decision'])
print('Bleu Done', flush=True)
results['meteor'] = meteor.compute(predictions=data[MODEL_NAME],references= data['Decision'])
print('Meteor Done', flush=True)
results['bertscore'] = bertscore.compute(predictions=data[MODEL_NAME],references= data['Decision'], lang='en', batch_size = 64)
print('BertScore Done', flush=True)
results['bartscore'] = {}
results['bartscore']['precision'] = bartscore.score(data['Decision'].tolist(), data[MODEL_NAME].tolist(), batch_size = 4)
results['bartscore']['recall'] = bartscore.score(data[MODEL_NAME].tolist(), data['Decision'].tolist(), batch_size = 4)

for i in range(len(results['bartscore']['precision'])):
    results['bartscore']['f1'] = results['bartscore']['recall'][i] + results['bartscore']['precision'][i] / 2
print('BartScore Done', flush=True)

cols = ['precision', 'recall', 'f1']

for c in cols:
    results['bertscore'][c] = pd.Series(results['bertscore'][c]).mean()
    results['bartscore'][c] = pd.Series(results['bartscore'][c]).mean()

results

# write results to a json file
import json
with open(f'../results/{MODEL_NAME}.json', 'w') as f:
    json.dump(results, f)


