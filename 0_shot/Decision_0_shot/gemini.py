import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
from time import time, sleep
import argparse
from tqdm import tqdm

def get_data(data: pd.DataFrame, max_length = -1):
    context = data['Context'].tolist()
    decision = data['Decision'].tolist()
    for i in range(len(context)):
        context[i] = f"This is an Architectural Decision Record. Provide a Decision for the Context given below.\n{context[i]}\n## Decision\n"
    if max_length != -1:
        removed = []
        context_new = []
        decision_new = []
        for i, (c, d) in enumerate(zip(context, decision)):
            if len(c) < max_length and len(d) < max_length:
                context_new.append(c)
                decision_new.append(d)
            else:
                removed.append(i)
        context = context_new
        decision = decision_new
        
        return context, decision, removed
    return context, decision, []


load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_KEY'))

model_name = "gemini-pro"

model = genai.GenerativeModel(model_name)

def run(start = 0, days = 1500):
    data = pd.read_csv('../ADR-data/context_decision.csv')
    # data = pd.read_csv('../results/gemini-fail.csv')

    context, decision, removed = get_data(data)

    predicted_decision = []
    new_context = []
    new_decision = []

    done = 0
    total = days - len(context[start:]) if days > len(context[start:]) else days

    for i, c in tqdm(enumerate(context[start:]), total=total):
        gen = None
        try:
            gen = model.generate_content(c)
            predicted_decision.append(gen._result.candidates[0].content.parts[0].text.replace("\n", "\\n"))
        except Exception as e:
            print(gen,c)
            print(e)
            print(f"Failed for {i}")
            predicted_decision.append("FAILED"*100)
        new_context.append(c.replace("\n", "\\n"))
        new_decision.append(decision[start + i].replace("\n", "\\n"))
        done += 1
        if done >= days:
            print("Done for the day")
            break
        sleep(5)

    new_decision = pd.DataFrame({'Context': new_context, 'Decision': new_decision, 'Prediction': predicted_decision})

    print(f"Prediction done for {len(predicted_decision)} records")
    # print(predicted_decision)

    df = pd.read_csv('../results/gemini.csv') if os.path.exists('../results/gemini.csv') else pd.DataFrame({'Context': [], 'Decision': [], 'Prediction': []})
    df = pd.concat([df, new_decision])
    df.to_csv('../results/gemini.csv', index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Gemini')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--day', type=int, default=1500, help='Number of records to predict in one day')
    args = parser.parse_args()
    
    run(args.start, args.day)