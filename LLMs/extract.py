import json
import pandas as pd
from bs4 import BeautifulSoup as bs

def extract(file_path: str) -> pd.DataFrame:
    data = json.load(open(file_path))
    df = pd.DataFrame(columns=["Rank", "Name", "Arena Score", "95% CI", "Votes", "Organization", "License", "Knowledge Cutoff"])
    for i, item in enumerate(data["data"]):
        df.loc[i] = [item[0], bs(item[1], 'html.parser').text, item[2], item[3], item[4], item[5], item[6], item[7]]
    return df
    
def main():
    df = extract('raw-data.json')
    df.to_csv('lmsys.csv', index=False)

if __name__ == '__main__':
    main()