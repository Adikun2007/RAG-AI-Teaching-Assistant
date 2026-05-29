import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", 
                    json={"model": "bge-m3",
                          "input": text_list})

    embedding = r.json()['embeddings']
    return embedding


df = joblib.load('chunks_embeddings.joblib')

incoming_query = input("Ask Anything: ")
query_embedding = create_embedding([incoming_query])[0]
# print(query_embedding)

# find similarity between query embedding and chunk embeddings
similarities = cosine_similarity([query_embedding], np.vstack(df['embedding'])).flatten()
# print(similarities)

top_results = 30
max_indices = similarities.argsort()[::-1][0:top_results]
# print(max_indices)

new_df = df.iloc[max_indices]
# print(new_df[['video_number', 'video_title', 'text']])

for index, row in new_df.iterrows():
    print(index, "video_number:", row['video_number'], "video_title:", row['video_title'], "text:", row['text'], "start:", row['start'], "end:", row['end'])