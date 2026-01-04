import os
import yaml
import faiss
import json
import requests
import tiktoken
import numpy as np
from datetime import date, datetime

# ---------------------------
# CONFIG
# ---------------------------
VAULT_PATH = "/home/ethan-silverthorne/Documents/Sync Vault/1 - Overview/Archive/2025/Quater 4/"
EMBED_MODEL = "nomic-embed-text"
FAISS_INDEX_PATH = "index.faiss"
META_PATH = "metadata.json"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
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
        chunks.append(detokenize(tokens[i:i + CHUNK_SIZE]))
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


# ---------------------------
# INDEXING
# ---------------------------
def build_index():
    vectors = []
    metadata = []

    for root, _, files in os.walk(VAULT_PATH):
        for file in files:
            if not file.endswith(".md"):
                continue

            path = os.path.join(root, file)
            meta, body = parse_markdown(path)

            if not body:
                continue

            chunks = chunk_text(body)

            for i, chunk in enumerate(chunks):
                vectors.append(embed(chunk))
                metadata.append({
                    "file": file,
                    "path": os.path.relpath(path, VAULT_PATH),
                    "date": normalize_metadata(meta.get("date")),
                    "day": normalize_metadata(meta.get("day")),
                    "chunk": i,
                    "text": chunk
                })

    if not vectors:
        raise RuntimeError("No vectors generated. Check vault path or note contents.")

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(vectors)} chunks.")

if __name__ == "__main__":
    build_index()

