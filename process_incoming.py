import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    })
    
    response = r.json()
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

import re

def clean_text(text):
    #  clean Bracket and quotes
    return re.sub(r'[\{\}\[\]"\'\(\)]', '', str(text))

# Ab saaf text lekar join karna
context_text = "\n".join([clean_text(t) for t in new_df["text"].tolist()])

prompt = f'''You are a helpful AI teaching assistant. 
Read the following transcript and answer the question ONLY using this text. If the answer is not in the transcript, say "I don't have information about this in the video subtitles." Do not guess or generate random text.

Transcript:
{context_text}

Question: "{incoming_query}"

Answer the question directly and clearly in simple English:'''
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

from groq import Groq

client = Groq(api_key="YOUR_API_KEY_HERE")

print("Cloud AI is thinking...")

# Groq sending request to Cloud API
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.1-8b-instant",
)

# Jo answer aaya usko nikalna
response = chat_completion.choices[0].message.content

print("\n--- ANSWER ---")
print(response)

# Answer ko txt file mein save karne ke liye
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)