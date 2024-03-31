import pandas as pd
import os
import requests

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

data = pd.read_csv('data.csv')

failed = []

ROOT = 'ADRs'
failed = open('failed.txt', 'w')
timeout = 10

git_url = 'https://api.github.com/repos/'

def wget(url):
    url = url.replace(" ", "%20")
    return os.system(f'wget --timeout={timeout} {url}')

for i, row in data.iterrows():
    # if row['URL'] != 'https://github.com/kconner/godspeed-you-blocked-developer.git':
    #     continue
    
    files = row['Names of ADR Files'].split('| ')
    
    url = row['URL'].replace('github.com', 'raw.githubusercontent.com')[:-4]
    branch = 'main'
    git_url = row['URL'].replace('github.com', 'api.github.com/repos')[:-4]

    folder = row["URL"].split("/")[-1][:-4]
    os.makedirs(f'{ROOT}/{folder}', exist_ok=True)
    for file in files:
        res = wget(f'{url}/{branch}/{file}')
        if res != 0:
            print("DOWNLOADING FROM MASTER")
            branch = 'master'
            res = wget(f'{url}/{branch}/{file}')
            if res != 0:
                print("GETTING BRANCH FROM API")
                headers = {
                    'Accept': 'application/vnd.github.v3+json',
                    'Authorization': 'token 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',
                }
                res = requests.get(f'{git_url}/branches', headers=headers)
                if res.status_code == 200:
                    branches = [b['name'] for b in res.json()]
                    res = 1
                    for b in branches:
                        print(f"NEW BRANCH NAME {b}")
                        res = wget(f'{url}/{b}/{file}')
                        if res == 0:
                            branch = b
                            break
                        
                    if res != 0:
                        print(f"{colors.FAIL}FAILED TO DOWNLOAD {file}{colors.ENDC}")
                        failed.write(f"{url}/{file}\n")
                        failed.flush()
                        continue
                else:
                    print(f"{colors.FAIL}FAILED TO DOWNLOAD {file} because of {res}{colors.ENDC}")
                    failed.write(f"{url}/{file}\n")
                    failed.flush()
                    break
        
        res = os.system(f'mv "{file.split("/")[-1]}" "{ROOT}/{folder}/{file.split("/")[-1]}"')
        if res != 0:
            print(f"Failed to move {file} to {ROOT}/{folder}")
            failed.write(f"{url}/{file}\n")
            failed.flush()
        