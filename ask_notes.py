import json
import faiss
import requests
import numpy as np
import re
import os
from ls import list_dir 

# ---------------------------
# CONFIG
# ---------------------------
LLM_MODEL = "qwen2.5:3b-instruct"
EMBED_MODEL = "nomic-embed-text"
FAISS_INDEX_PATH = "index.faiss"
DOCUMENT_FAISS_INDEX_PATH = "documents_index.faiss"
META_PATH = "metadata.json"
DOCUMENT_META_PATH = "documents_metadata.json"
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
# Workouts
# ---------------------------
def ask_workouts(question):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    allowed_indices = [
        i for i, m in enumerate(metadata)
        if m.get("type") == "workouts"
    ]

    # ---- 1. Exact-date lookup ----
    query_date = extract_iso_date(question)
    if query_date:
        context = []
        print("Exact-date lookup:", query_date)

        for m in metadata:
            if m.get("type") == "workouts" and m.get("date") == query_date:
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

    # ---- 2. Week-based lookup (single or multiple weeks) ----def ask_document(path, question):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = f"""
You are answering questions about a single document.

Rules:
- Use ONLY the document content below.
- Do NOT invent information.
- If the answer is not present, say so.
- Preserve exact wording when relevant.

Document:
<<<
{content}
>>>

Question:
{question}
"""
    weeks = extract_weeks(question)

    if weeks:
        week_context = {}

        for week in weeks:
            week_context[week] = []
            print(f"Week-based lookup: Week {week}")

            for m in metadata:
                path = m.get("path", "")
                if (m.get("type") == "workouts" and f"Week {week}" in path):
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
        if idx not in allowed_indicies:
            continue
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

#########################
### workout summaries
#########################

def ask_workout_summaries(question):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    
    allowed_indices = [
        i for i, m in enumerate(metadata)
        if m.get("type") == "workouts-summary"
    ] 

    # ---- 2. Week-based lookup (single or multiple weeks) ----
    weeks = extract_weeks(question)

    if weeks:
        week_context = {}

        for week in weeks:
            week_context[week] = []
            print(f"Week-based lookup: Week {week}")

            for m in metadata:
                path = m.get("path", "")
                if (m.get("type") == "workouts-summary" and f"Week {week}" in path):
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
        if idx not in allowed_indices:
            continue
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

##################################
######    Schedule
##################################

def ask_schedule(question):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    allowed_indices = [
        i for i, m in enumerate(metadata)
        if m.get("type") == "schedule"
    ]

    # ---- 1. Exact-date lookup ----
    query_date = extract_iso_date(question)
    if query_date:
        context = []
        print("Exact-date lookup:", query_date)

        for m in metadata:
            if m.get("type") == "schedule" and m.get("date") == query_date:
                context.append(
                f"[Date: {m.get('date')}, File: {m.get('file')}]\n{m['text']}"
                )
                print(m["file"], m["date"], m.get("path", ""))

        if not context:
            print("Not found in notes.")
            return

        prompt = f"""
You are answering questions about a day on a personal schedule

The context below is a raw schedule log
Summarize what was done for the day. Do not interrupt. Make sure the times match exactly what was given in the document

Context:
{chr(10).join(context)}

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
        if idx not in allowed_indicies:
            continue
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

#############
## Document
############
def ask_document(document, question):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # print(content) debugging
    prompt = f"""
    You are answering questions about a single document.

    Rules:
    - Use ONLY the document content below.
    - Do NOT invent information.
    - If the answer is not present, say so.
    - Preserve exact wording when relevant.

    Document:
    {content}

    Question: {question}
    """ 
    print("\nAnwser:\n", chat(prompt))

if __name__ == "__main__":
    import sys
    while True:
        try:
            choice = int(input("Is your query regarding\n1. Workouts\n2. Schedule\n3. Workout Weekly Summaries\n4. Documents\n"))
            if choice == 1:
                prompt = input("What do you want to know")
                ask_workouts(prompt)
            elif choice == 2:
                prompt = input("What do you want to know")
                ask_schedule(prompt)
            elif choice == 3:
                prompt = input("What do you want to know")
                ask_workout_summaries(prompt)
            elif choice == 4:
                files = list_dir("/home/ethan-silverthorne/Documents/Sync Vault/4 - Documents")
                for i, f in enumerate(files):
                    print(f"{i}: {os.path.basename(f)}")

                selection = int(input("Select a document by number: "))
                path = files[selection]
                print(f"You chose to analyze {path}")

                prompt = input("What do you want to know? ")
                ask_document(path, prompt)

        except ValueError:
            print("Must be a valid number")

