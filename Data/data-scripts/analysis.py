import os
import csv
from pprint import pprint
from transformers import AutoTokenizer
import pandas as pd

x = ['Context', 'Context and Problem Statement', 'Decision Drivers', 'Decision Drivers <!-- optional -->', 'Pros and Cons of the Options', 'Problem', 'Pros and Cons of the Options <!-- optional -->']
y = ['Decision', 'Decision Outcome', 'Decisions']
CACHE_DIR = '/scratch/llm4adr/cache'

def extract(file):
    lines = file.readlines()
    context = ''
    context_on = False
    decision = ''
    decision_on = False
    reached_decision = False
    for line in lines:
        if '##' in line and '###' not in line:
            decision_on = False
            context_on = False
        # if any of the elements in x is in line, then context_on = True
        if any([ele in line for ele in x]) and not reached_decision:
            context_on = True
            # continue
        if decision_on and line.strip() != '':
            decision += line.strip() + '\\n'
        if any([ele in line for ele in y]):
            decision_on = True
            reached_decision = True
            # continue
        if context_on and line.strip() != '':
            context += line.strip() + '\\n'
    return context, decision

def get_context_decision(parent_dir, output_file='../context_decision.csv'):
    writer = csv.writer(open(output_file, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['File Name','Context', 'Decision'])
    # get all folders in the parent directory
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    for folder in folders:
        files = os.listdir(os.path.join(parent_dir, folder))
        for file in files:
            file_path = os.path.join(parent_dir, folder, file)
            with open(file_path, 'r') as f:
                context, decision = extract(f)
                if context == '' or decision == '':
                    continue
                writer.writerow([f'{folder}/{file}',context, decision])
        
def get_headings(parent_dir, output_file='../headings.csv'):
    headings = {}
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    for folder in folders:
        files = os.listdir(os.path.join(parent_dir, folder))
        for file in files:
            file_path = os.path.join(parent_dir, folder, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    sline = line.strip()
                    if sline.startswith('## '):
                        if sline not in headings:
                            headings[sline] = 1
                        else:
                            headings[sline] += 1
    # print headings sorted by count
    sorted_headings = sorted(headings.items(), key=lambda x: x[1], reverse=True)
    # write sorted headings to file
    writer = csv.writer(open(output_file, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Heading', 'Count'])
    for heading, count in sorted_headings[:20]:
        writer.writerow([heading, count])
        
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", cache_dir=CACHE_DIR)
        
def count_tokens(text):
    tokens = tokenizer(text, truncation=False)['input_ids']
    return len(tokens)
        
def get_count(num_tokens: int) -> int:
    data = pd.read_csv('../ADR-data/context_decision.csv')
    print(len(data), end=' ')
    
    data['total'] = data['Context'] + data['Decision']
    # print(data.iloc[0])
    data['num_tokens'] = data['total'].apply(count_tokens)
    return len(data[data['num_tokens'] <= num_tokens])

def main():
    # get_headings('./done_ADRs')
    # get_context_decision('../../done_ADRs')
    print(get_count(500))
    print(get_count(1000))
    print(get_count(2000))
    print(get_count(4000))
    print(get_count(8000))
    
if __name__ == '__main__':
    main()
    