import openai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
# model = "babbage-002"
max_tokens = 1000
# openai.api_key = os.getenv("OPENAI_KEY")

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_KEY"),
)


def main():
    df = pd.read_csv("context_decision.csv")
    # take a subset of 100 samples
    df = df.sample(1)
    decisions = []

    for index, row in df.iterrows():
        context = row["Context"]
        print(context)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a software architect. You will help people in taking software architectural decisions.
                    You need to take a decision based on a given context. Also mention all the options you considered while making this decision.
                    """,
                },
                {"role": "user", "content": context},
            ],
            # prompt=f"""You are a software architect. You will help people in taking software architectural decisions.
            # You need to take a decision based on a given context. Also mention all the options you considered while making this decision.
            # The context is:
            # {context}
            # The decision taken is:
            # We have decided to go with """,
        )
        # print(response.choices)
        # decisions.append(response.choices[0].text)
        decisions.append(response.choices[0].message.content)

    print(decisions[0])
    # df["decision"] = decisions
    # df.to_csv("context_decision_llm.csv", index=False)


if __name__ == "__main__":
    main()
