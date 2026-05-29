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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", 
                    json={"model": "deepseek-r1:32b",
                          "prompt": prompt,
                          "stream": False})

    response = r.json()['response']
    return response

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

prompt = f'''You are an expert learning guide for a curated video course on Data Science, Machine Learning, and AI careers. You have access to transcripts from exactly 15 videos:

1. What is Data Science + Roadmap
2. Data Analyst vs Data Scientist vs Data Engineer (Roles Comparison)
3. Machine Learning Introduction
4. Machine Learning Complete Roadmap
5. AI Engineer Roadmap 2026
6. Geospatial Data Scientist
7. 14 ML Projects for Internship
8. How to Get a Job in Data Science
9. Get a Job IMMEDIATELY as Data Analyst
10. Data Analyst Portfolio 2026
11. ML Resume that got 5 Interviews
12. ATS Friendly Resume
13. Perfect Data Science Resume (Google)
14. SQL Interview Questions
15. Behavioural Interview - STAR Technique

You will be given the top {top_results} most semantically relevant transcript chunks to answer the user's question. Each chunk contains: video_number, video_title, start timestamp (seconds), end timestamp (seconds), and text.

Here are the relevant chunks:
{new_df[['video_number', 'video_title', 'start', 'end', 'text']].to_json(orient='records')}

---

User Question: "{incoming_query}"

---

INSTRUCTIONS:

1. TOPIC CHECK: If the question is NOT related to data science, machine learning, AI, Python, career advice, resumes, job hunting, SQL, or interviews — respond ONLY with:
   "I can only answer questions related to Data Science, Machine Learning, AI careers, Python, SQL, Resumes, and Interview Preparation. Please ask something related to these topics!"

2. If the question IS relevant, follow this response structure:

## Answer
Give a clear, concise, and helpful answer based ONLY on the provided chunks. Do not hallucinate or add information not present in the chunks.

## Where to Learn This
For each relevant video, list:
- 📹 **Video [number]: [title]**
  - 📍 Timestamp: [start]s – [end]s (or convert to MM:SS format)
  - 💡 What is covered at this point: [brief description from the chunk text]

## Recommended Watch Order
If multiple videos are relevant, suggest the best order to watch them for maximum learning.

## Pro Tip (optional)
If you can add a genuinely useful insight from the content, add it here.

---

RULES:
- Base your answer strictly on the chunk content provided.
- Always mention specific video numbers and timestamps so the user can navigate directly.
- Convert seconds to MM:SS format for readability (e.g., 125s → 2:05).
- If only one video is relevant, skip the "Recommended Watch Order" section.
- Be conversational but precise.
- Do not repeat the same chunk information multiple times.
'''

with open('prompt.txt', 'w') as f:
    f.write(prompt)


response = inference(prompt)['response']
print(response)
with open('response.txt', 'w') as f:
    f.write(response)

for index, row in new_df.iterrows():
    print(index, "video_number:", row['video_number'], "video_title:", row['video_title'], "text:", row['text'], "start:", row['start'], "end:", row['end'])