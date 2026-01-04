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


# ---------------------------
# QUERYING
# ---------------------------
def ask(question):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    context = []

    query_date = extract_iso_date(question)

    # ---- Exact-date lookup ----
    if query_date:
        print("Exact-date lookup:", query_date)
        for m in metadata:
            if m.get("date") == query_date:
                context.append(m["text"])
                print(m["file"], m["date"])

        if not context:
            print("Not found in notes.")
            return

    # ---- Semantic fallback ----
    else:
        q_vec = embed(question)
        D, I = index.search(
            np.array([q_vec]).astype("float32"),
            TOP_K
        )

        print("Semantic retrieval:")
        for idx in I[0]:
            m = metadata[idx]
            context.append(m["text"])
            print(m["file"], m["date"])

        if not context:
            print("Not found in notes.")
            return

    prompt = f"""
You are answering questions about personal workout logs.

The context below is a raw workout log for a single day.
If context is provided, you MUST answer by interpreting the log.
Do NOT say "Not found in notes" unless the context is empty.

Describe what workout was performed and which muscle groups were trained.
Do not invent information.

Context:
{chr(10).join(context)}

Question:
{question}
"""

    answer = chat(prompt)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    import sys
    ask(" ".join(sys.argv[1:]))

