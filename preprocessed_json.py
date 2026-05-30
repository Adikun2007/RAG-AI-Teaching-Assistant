import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", 
                    json={"model": "bge-m3",
                          "input": text_list})

    embedding = r.json()['embeddings']
    return embedding

# store = create_embedding("What is data science?")
# print(store)


merged_jsons = os.listdir('merged_jsons')
my_dicts = []
chunk_id = 0

for json_file in merged_jsons:
    with open(f"merged_jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating embeddings for Json file: {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]

        chunk["video_number"] = chunk.get("video_number") or content.get("video_number")
        chunk["video_title"] = chunk.get("video_title") or content.get("video_title")
        chunk["source_file"] = chunk.get("source_file") or content.get("source_file") or json_file

        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
# print(df)

# Save the dataframe to a CSV file
joblib.dump(df, 'chunks_embeddings.joblib')
