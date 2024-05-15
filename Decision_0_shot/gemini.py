import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
from time import time, sleep
import argparse

def get_data(data: pd.DataFrame, max_length = -1):
    context = data['Context'].tolist()
    decision = data['Decision'].tolist()
    for i in range(len(context)):
        context[i] = f"This is an Architectural Decision Record. Provide a Decision for the Context given below.\n{context[i]}\nDecision:"
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

def run(start = 0, days = 1500, minutes = 15):
    data = pd.read_csv('../ADR-data/context_decision.csv')

    context, decision, removed = get_data(data)

    predicted_decision = []
    new_context = []
    new_decision = []

    done = 0
    minute = 0
    t = 0

    for i, c in enumerate(context[start:]):
        s = time()
        predicted_decision.append(model.generate_content(c)._result.candidates[0].content.parts[0].text.replace("\n", "\\n"))
        new_context.append(c.replace("\n", "\\n"))
        new_decision.append(decision[start + i].replace("\n", "\\n"))
        dur = time() - s
        t += dur
        done += 1
        minute += 1
        if done >= days:
            print("Done for the day")
            break
        if t >= 60:
            t = 0
            minute = 0
        if minute >= minutes:
            print(f"Taking a break for {60 - t + 1} seconds")
            sleep(60 - t + 1)
            minute = 0

    new_decision = pd.DataFrame({'Context': new_context, 'Decision': new_decision, 'Prediction': predicted_decision})

    print(f"Prediction done for {len(predicted_decision)} records")
    print(predicted_decision)

    df = pd.read_csv('../results/gemini.csv') if os.path.exists('../results/gemini.csv') else pd.DataFrame({'Context': [], 'Decision': [], 'Prediction': []})
    df = pd.concat([df, new_decision])
    df.to_csv('../results/gemini.csv', index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Gemini')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--day', type=int, default=1500, help='Number of records to predict in one day')
    parser.add_argument('--minute', type=int, default=15, help='Number of records to predict in one minute')
    args = parser.parse_args()
    
    run(args.start, args.day, args.minute)