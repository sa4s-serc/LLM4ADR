from openai import OpenAI
import openai
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
        context[i] = f"{context[i]}\n## Decision\n"
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

system_message = "This is an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User."

client = OpenAI(api_key = os.getenv('OPENAI_KEY'))

# MODEL_NAME = "gpt-4-turbo-preview"
MODEL_NAME = "babbage-002"

def run(start = 0, days = 10000):
    data = pd.read_csv('../ADR-data/context_decision.csv')

    context, decision, removed = get_data(data)

    predicted_decision = []
    new_context = []
    new_decision = []

    done = 0
    total = len(context[start:]) if days > len(context[start:]) else days

    for i, c in tqdm(enumerate(context[start:]), total=total):
        response = None
        try:
            # gpt-3.5 an 4
            # response = client.chat.completions.create(
            #     model=MODEL_NAME,
            #     messages=[
            #         {
            #         "role": "system",
            #         "content": system_message
            #         },
            #         {
            #         "role": "user",
            #         "content": c
            #         }
            #     ],
            #     temperature=1,
            #     max_tokens=2500,
            #     top_p=1,
            #     frequency_penalty=0,
            #     presence_penalty=0
            # )
            # predicted_decision.append(response.choices[0].message.content.replace("\n", "\\n"))
            # babbage-002
            response = client.completions.create(
                model=MODEL_NAME,
                prompt=f"{system_message}\n{c}",
                temperature=0.5,
                max_tokens=2500,
                top_p=1,
                frequency_penalty=0.4,
                presence_penalty=0
            )
            predicted_decision.append(response.choices[0].text.replace("\n", "\\n"))
        except Exception as e:
            print(response,c)
            print(e)
            print(f"Failed for {i}")
            predicted_decision.append("FAILED"*100)
        new_context.append(c.replace("\n", "\\n"))
        new_decision.append(decision[start + i].replace("\n", "\\n"))
        done += 1
        if done >= days:
            print("Done for the day")
            break

    new_decisions = pd.DataFrame({'Context': new_context, 'Decision': new_decision, 'Prediction': predicted_decision})

    print(f"Prediction done for {len(predicted_decision)} records")
    # print(predicted_decision)

    new_decisions.to_csv(f'../results/{MODEL_NAME}.csv', index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Gemini')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--day', type=int, default=10000, help='Number of days to run')
    args = parser.parse_args()
    
    run(args.start, args.day)