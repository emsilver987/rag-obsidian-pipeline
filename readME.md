# This project was made to demonsrate my ability to construct and use RAG pipelines

<img width="1910" height="484" alt="image" src="https://github.com/user-attachments/assets/04487c5e-6d83-47a6-8fe7-7546caa2af3f" />

<img width="1235" height="145" alt="Screenshot from 2026-01-05 22-53-33" src="https://github.com/user-attachments/assets/50066d62-827d-4c84-a239-d4ea8e232b27" />

## Tools and Technologies
nomic-embed-text model is used to index notes

FAISS also used for indexing

OLLAMA serve running in background to handle embedding and query requests
qwen2.5:3b-instruct is used as the main LLM when querying

Everything here runs entirely locally

### Instructions

ollama serve needed in seperate terminal, can see logs there

ollama -ps to check procceses, if CPU overloaded can kill some

To index notes: index_notes.py or index_documents.py
- This takes YAML from each vault and writes to metadata.json (or documents_metadata.json)
- Builds index.faiss(or documents_index.faiss)

To ask questions: python ask_notes.py 
- Follow CLI usage guide below

To classifly splits: python classify_split.py
- Indexing and metadata enhancement with LLM

Will eventually want functionaility to be able to index without changing the hardcoded path

## CLI Usage

You are prompted with some options of how you would like to use the vault
1. Workouts
2. Schedule
3. Workout Weekly Summaries
4. Documents

Each of these work in a different ways both in how they do RAG and how they prompt the agent

Workouts
- Looks for ISO date match
- Looks for week match
- falls back on semantic reasoning

Schedule
- looks for ISO dates
- No fall back, if date was not correct/ISO format, no document found

Workout Weekly Summaries
- look for week
- if >1 weeks does comparision logic with different prompt

Documents
- prompts user with a list of all documents numbered in given directory
- user selects document they want to query
- user asks their question

## Future
I think it would be cool to eventually make this code more accesible to people who use obsidian vault
It currenlty is intended only for my machine but it would be awesome to allow others to easily use it


