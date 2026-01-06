# This project was made to demonsrate my ability to construct and use RAG pipelines

<img width="1910" height="484" alt="image" src="https://github.com/user-attachments/assets/04487c5e-6d83-47a6-8fe7-7546caa2af3f" />

## Tools and Technologies
nomic-embed-text model is used to index notes

FAISS also used for indexing

OLLAMA serve running in background to handle embedding and query requests

qwen2.5:3b-instruct is used as the main LLM when querying

Everything here runs entirely locally

## Future
I think it would be cool to eventually make this code more accesible to people who use obsidian vault
It currenlty is intended only for my machine but it would be awesome to allow others to easily use it

### Instructions

ollama serve needed in seperate terminal, can see logs there

ollama -ps to check procceses, if CPU overloaded can kill some

To index notes: python rag_obsidian.py index
- This takes YAML from each vault and writes to metadata.json

To ask questions: python rag_obsidian.py ask "What was I doing on December 13 2025?"
- currently has exact ISO format date matching, week matching, and last case sematic reasoning (unstable)

To classifly splits: python classify_split.py
- This is to add additional metadata to our indexed files without writing to the files themselves

Will eventually want functionaility to be able to index without changing the hardcoded path

