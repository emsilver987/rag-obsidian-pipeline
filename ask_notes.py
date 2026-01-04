import json
import faiss
import requests
import numpy as np
import re

# ---------------------------
# CONFIG
# ---------------------------
LLM_MODEL = "qwen2.5:3b-instruct"
EMBED_MODEL = "nomic-embed-text"
FAISS_INDEX_PATH = "index.faiss"
META_PATH = "metadata.json"
TOP_K = 5
OLLAMA_URL = "http://localhost:11434"

# ---------------------------
# UTILS
# ---------------------------
def embed(text):
    res = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return res.json()["embedding"]


def chat(prompt):
    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return res.json()["response"]


def extract_iso_date(text):
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else None


def extract_weeks(text):
    return re.findall(r"week\s*(\d+)", text, re.IGNORECASE)


# ---------------------------
# QUERYING
# ---------------------------
def ask(question):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    # ---- 1. Exact-date lookup ----
    query_date = extract_iso_date(question)
    if query_date:
        context = []
        print("Exact-date lookup:", query_date)

        for m in metadata:
            if m.get("date") == query_date:
                context.append(m["text"])
                print(m["file"], m["date"], m.get("path", ""))

        if not context:
            print("Not found in notes.")
            return

        prompt = f"""
You are answering questions about personal workout logs.

The context below is a raw workout log for a single day.
Interpret it directly.

Context:
{chr(10).join(context)}

Question:
{question}
"""
        print("\nAnswer:\n", chat(prompt))
        return

    # ---- 2. Week-based lookup (single or multiple weeks) ----
    weeks = extract_weeks(question)

    if weeks:
        week_context = {}

        for week in weeks:
            week_context[week] = []
            print(f"Week-based lookup: Week {week}")

            for m in metadata:
                path = m.get("path", "")
                if f"Week {week}" in path:
                    week_context[week].append(m["text"])
                    print(m["file"], m["date"], path)

        if not any(week_context.values()):
            print("Not found in notes.")
            return

        # ---- Multi-week comparison ----
        if len(weeks) > 1:
            prompt = f"""
You are comparing workout activity across multiple weeks.

Each section below contains raw workout logs grouped by week.
Compare training volume, exercise focus, and muscle groups trained.
Do NOT invent information.

{"".join(
    f"\nWeek {week}:\n" + chr(10).join(week_context[week])
    for week in weeks
)}

Question:
{question}
"""
            print("\nAnswer:\n", chat(prompt))
            return

        # ---- Single week summary ----
        single_week = weeks[0]
        prompt = f"""
You are answering questions about personal workout logs.

The context below contains all workouts performed in Week {single_week}.
Summarize what was done and which muscle groups were trained.

Context:
{chr(10).join(week_context[single_week])}

Question:
{question}
"""
        print("\nAnswer:\n", chat(prompt))
        return

    # ---- 3. Semantic fallback (last resort) ----
    context = []
    q_vec = embed(question)
    D, I = index.search(
        np.array([q_vec]).astype("float32"),
        TOP_K
    )

    print("Semantic retrieval:")
    for idx in I[0]:
        m = metadata[idx]
        context.append(m["text"])
        print(m["file"], m["date"], m.get("path", ""))

    if not context:
        print("Not found in notes.")
        return

    prompt = f"""
You are answering questions about personal workout logs.

Interpret the context below and answer the question.
Do NOT invent information.

Context:
{chr(10).join(context)}

Question:
{question}
"""
    print("\nAnswer:\n", chat(prompt))


if __name__ == "__main__":
    import sys
    ask(" ".join(sys.argv[1:]))

