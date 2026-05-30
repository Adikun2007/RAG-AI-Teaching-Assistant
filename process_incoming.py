import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

from config import GROQ_API_KEY


EMBED_MODEL = "bge-m3"
GROQ_MODEL = "llama-3.3-70b-versatile"

EMBED_URL = "http://localhost:11434/api/embed"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DB_PATH = Path("chunks_embeddings.joblib")
PROMPT_LOG_PATH = Path("prompt.txt")
RESPONSE_LOG_PATH = Path("response.txt")

TOP_RESULTS = 8
THRESHOLD = 0.42
SHOW_DEBUG_SOURCES = False


ALLOWED_TOPICS = [
    "data science", "machine learning", "artificial intelligence", "ai",
    "deep learning", "python", "sql", "data analyst", "data scientist",
    "ai engineer", "resume", "interview", "job", "career", "roadmap",
    "statistics", "pandas", "numpy", "model", "algorithm"
]


def create_embedding(text_list):
    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text_list},
        timeout=60
    )
    response.raise_for_status()
    data = response.json()

    if "embeddings" not in data:
        raise ValueError(f"Ollama embedding error: {data}")

    return data["embeddings"]


def is_allowed_question(question):
    q = question.lower()
    return any(topic in q for topic in ALLOWED_TOPICS)


def format_time(seconds):
    seconds = int(float(seconds or 0))
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"


def clean_title(row):
    title = row.get("video_title")

    if pd.isna(title) or title in [None, "", "None"]:
        title = row.get("source_file")

    if pd.isna(title) or title in [None, "", "None"]:
        title = "Best matching video"

    return str(title)


def retrieve_chunks(df, query):
    query_embedding = create_embedding([query])[0]
    chunk_embeddings = np.vstack(df["embedding"].values)

    similarities = cosine_similarity([query_embedding], chunk_embeddings).flatten()

    valid_indices = np.where(similarities >= THRESHOLD)[0]

    if len(valid_indices) == 0:
        return pd.DataFrame(), similarities

    sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
    top_indices = sorted_indices[:TOP_RESULTS]

    results = df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]
    results = results.drop_duplicates(subset=["text"])

    return results, similarities


def build_context(chunks_df):
    rows = []

    for _, row in chunks_df.iterrows():
        rows.append({
            "video_title": clean_title(row),
            "start": format_time(row.get("start", 0)),
            "end": format_time(min(float(row.get("end", 0)), float(row.get("start", 0)) + 120)),
            "text": str(row.get("text", ""))[:900]
        })

    return json.dumps(rows, ensure_ascii=False, indent=2)


def ask_groq(question, context):
    system_prompt = """
You are a helpful Data Science, AI, and career learning assistant.

Your style:
- Explain like a clear teacher, similar to ChatGPT or Claude.
- Use simple beginner-friendly language.
- Give broad but useful answers.
- Use headings and bullet points when helpful.
- Use examples to make concepts easy.
- Use ONLY the provided transcript context when possible.
- If the context is limited, give a safe general explanation related to the topic.
- Do not mention transcript, chunks, retrieval, or context.
"""

    user_prompt = f"""
Transcript context:
{context}

User question:
{question}

Write the answer in this format:

Answer:
Give a clear explanation in 2-4 short paragraphs.

Include these when useful:
- Simple definition
- Why it matters
- Real-world examples
- Main components or skills
- Career or learning path if relevant

Next step:
- Give one practical next step for a beginner.

Rules:
- Target length: 180 to 350 words.
- Use bullet points if they improve clarity.
- Do not give a one-line answer.
- Do not be too short.
- Do not sound robotic.
- Do not repeat the same point.
- Do not mention the transcript or context.
- Do not recommend videos.
"""

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 700,
        },
        timeout=60
    )

    response_json = response.json()

    if "choices" not in response_json:
        print("\nGroq API Error:")
        print(response_json)
        error_info = response_json.get("error", {})
        raise ValueError(error_info.get("message", response_json))

    PROMPT_LOG_PATH.write_text(user_prompt, encoding="utf-8")

    answer = response_json["choices"][0]["message"]["content"].strip()
    RESPONSE_LOG_PATH.write_text(answer, encoding="utf-8")

    return answer


def print_best_video(chunks_df):
    if chunks_df.empty:
        print("\nBest video:")
        print("- No strong video match found")
        return

    best = chunks_df.iloc[0]
    title = clean_title(best)

    start = float(best.get("start", 0))
    end = float(best.get("end", start + 60))
    end = min(end, start + 120)

    print("\nBest video:")
    print(f"- {title}")
    print(f"- {format_time(start)} -> {format_time(end)}")


def main():
    if not DB_PATH.exists():
        print("chunks_embeddings.joblib not found. Run your embedding script first.")
        return

    df = joblib.load(DB_PATH)

    incoming_query = input("Ask Anything: ").strip()

    if not incoming_query:
        print("Please ask a question.")
        return

    if not is_allowed_question(incoming_query):
        print("Answer:")
        print("- I only answer Data Science, AI, Python, SQL, resume, and career questions.")
        return

    chunks_df, _ = retrieve_chunks(df, incoming_query)

    if chunks_df.empty:
        print("Answer:")
        print("- I could not find useful content for this question.")
        print("\nNext step:")
        print("- Try asking with simpler or more specific words.")
        return

    context = build_context(chunks_df)
    response = ask_groq(incoming_query, context)

    print()
    print(response)
    print_best_video(chunks_df)

    if SHOW_DEBUG_SOURCES:
        print("\nDebug sources:")
        for index, row in chunks_df.iterrows():
            print(
                index,
                "video_number:", row.get("video_number"),
                "video_title:", row.get("video_title"),
                "start:", row.get("start"),
                "end:", row.get("end"),
                "similarity:", round(row.get("similarity", 0), 3),
            )


if __name__ == "__main__":
    main()