import os
import yaml
import faiss
import json
import requests
import tiktoken
import numpy as np
import re
from datetime import date, datetime

# ---------------------------
# CONFIG
# ---------------------------
VAULT_PATH = "/home/ethan-silverthorne/Documents/Sync Vault/1 - Overview/Archive/2025/Quater 4/Week 13"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b-instruct"
FAISS_INDEX_PATH = "index.faiss"
META_PATH = "metadata.json"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5

OLLAMA_URL = "http://localhost:11434"


# ---------------------------
# UTILS
# ---------------------------

def normalize_metadata(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value

def tokenize(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)


def detokenize(tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)


def chunk_text(text):
    tokens = tokenize(text)
    chunks = []

    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = tokens[i:i + CHUNK_SIZE]
        chunks.append(detokenize(chunk))

    return chunks


def parse_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if content.startswith("---"):
        _, fm, body = content.split("---", 2)
        meta = yaml.safe_load(fm)
    else:
        meta = {}
        body = content

    return meta or {}, body.strip()


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
# INDEXING
# ---------------------------
def build_index():
    vectors = []
    metadata = []

    for file in os.listdir(VAULT_PATH):
        if not file.endswith(".md"):
            continue

        path = os.path.join(VAULT_PATH, file)
        meta, body = parse_markdown(path)

        chunks = chunk_text(body)

        for i, chunk in enumerate(chunks):
            vec = embed(chunk)
            vectors.append(vec)
            metadata.append({
    "file": file,
    "date": normalize_metadata(meta.get("date")),
    "day": normalize_metadata(meta.get("day")),
    "chunk": i,
    "text": chunk
})


    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(vectors)} chunks.")


# ---------------------------
# QUERYING
# ---------------------------


def ask(question):
    # Load index + metadata
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    context = []  # <-- DEFINE ONCE, ALWAYS EXISTS

    # ---- Step 1: Check for exact date ----
    query_date = extract_iso_date(question)

    if query_date:
        print("Exact-date lookup:", query_date)
        for m in metadata:
            if m.get("date") == query_date:
                context.append(m["text"])
                print(m["file"], m["date"])

        if not context:
            print("Not found in notes.")
            return

    # ---- Step 2: Semantic fallback ----
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

    # ---- Step 3: Ask the LLM (context ALWAYS exists now) ----

    
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
# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag_obsidian.py index")
        print("  python rag_obsidian.py ask \"your question\"")
        sys.exit(1)

    if sys.argv[1] == "index":
        build_index()
    elif sys.argv[1] == "ask":
        ask(" ".join(sys.argv[2:]))

