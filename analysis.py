import os
import csv
from pprint import pprint

x = ['Context', 'Context and Problem Statement', 'Decision Drivers', 'Decision Drivers <!-- optional -->', 'Pros and Cons of the Options', 'Problem', 'Pros and Cons of the Options <!-- optional -->']
y = ['Decision', 'Decision Outcome', 'Decisions']

def extract(file):
    lines = file.readlines()
    context = ''
    context_on = False
    decision = ''
    decision_on = False
    for line in lines:
        if '##' in line:
            decision_on = False
            context_on = False
        # if any of the elements in x is in line, then context_on = True
        if any([ele in line for ele in x]):
            context_on = True
            continue
        if any([ele in line for ele in y]):
            decision_on = True
            continue
        if context_on and line.strip() != '':
            context += line.strip() + '\\n'
        if decision_on and line.strip() != '':
            decision += line.strip() + '\\n'
    return context, decision

def get_context_decision(parent_dir, output_file='context_decision.tsv'):
    writer = csv.writer(open(output_file, 'w'), delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['File Name','Context', 'Decision'])
    # get all folders in the parent directory
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    for folder in folders:
        files = os.listdir(os.path.join(parent_dir, folder))
        for file in files:
            file_path = os.path.join(parent_dir, folder, file)
            with open(file_path, 'r') as f:
                context, decision = extract(f)
                # print('Context:', context)
                # print('Decision:', decision)
                if context == '' or decision == '':
                    # print(f'Error in {file_path}')
                    continue
                writer.writerow([f'{folder}/{file}',context, decision])
            # break
        # break
        
def get_headings(parent_dir, output_file='headings.tsv'):
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
    writer = csv.writer(open(output_file, 'w'), delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Heading', 'Count'])
    for heading, count in sorted_headings[:20]:
        writer.writerow([heading, count])
    
def main():
    # get_headings('./done_ADRs')
    get_context_decision('./done_ADRs')
    
if __name__ == '__main__':
    main()
    