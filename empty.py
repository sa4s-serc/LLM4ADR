# find out all folders which are empty in a given directory
# Usage: python empty.py <directory>

import os
import sys
import pandas as pd

def empty_folders(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        if not subfolders and not filenames:
            print(foldername)
            
def count_files(directory):
    count = 0
    for foldername, subfolders, filenames in os.walk(directory):
        count += len(filenames)
    return count

def count_folders(directory):
    count = 0
    for foldername, subfolders, filenames in os.walk(directory):
        count += len(subfolders)
    return count

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python empty.py <directory>')
    else:
        empty_folders(sys.argv[1])
        print('Total files:', count_files(sys.argv[1]))
        print('Total folders:', count_folders(sys.argv[1]))
        
# if __name__ == '__main__':
#     df = pd.read_csv('data.csv')
#     total = 0
#     for i, row in df.iterrows():
#         adrs = row['Number of ADR Files']
#         total += adrs
#     print(total)