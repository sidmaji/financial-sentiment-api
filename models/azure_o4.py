import os

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
model_name = os.getenv("AZURE_MODEL_NAME")
deployment = os.getenv("AZURE_DEPLOYMENT")

subscription_key = "BxuJJgNPvAtzbDvzhcoddaeUIV9k1VmGUXc20hpf5x19PePAwwawJQQJ99BFAC4f1cMXJ3w3AAAAACOGLDtJ"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

preds = []

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

for i in range(len(val_df["sentence"].tolist())):
    print(i)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a financial sentiment classifier. Respond with one word: neutral, positive, or negative.",
            },
            {
                "role": "user",
                "content": f"Classify the sentiment of this sentence as positive, neutral, or negative:\n\n{val_df['sentence'].tolist()[i]}",
            },
        ],
        max_completion_tokens=10000,
        model=deployment,
    )

    preds.append(response.choices[0].message.content)

    print("o4-mini:", response.choices[0].message.content)
    print("validation:", val_df["label"].tolist()[i])
    print()
