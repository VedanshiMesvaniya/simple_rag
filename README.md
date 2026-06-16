
# Simple Local RAG POC

This is a small local RAG proof of concept based on the working reference project.

What it does:
- ingests `PDF` files only
- stores vectors in local Chroma
- tracks incremental processing in `vector_db/processed_files.json`
- answers from retrieved context only
- logs chat and ingestion activity in `logs/`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Ollama and pull the models:

```bash
ollama serve
ollama pull bge-m3:latest
ollama pull llama3.1:8b
```

4. Make directory `data/` and Put your PDF files in `data/`.

## Run

Launch the Streamlit UI:

```bash
streamlit run app.py
```

Or run ingestion only:

```bash
python ingest.py
```

Or run chat directly:

```bash
python rag_chat.py
```

## Config

Change models and retrieval settings in `config.py` only.

## Notes

- The system answers only from the retrieved document context.
- If the answer is missing, it responds with:

```text
I don't know based on the available documents.
```

## UI

The Streamlit interface includes:

- a dark two-panel workspace
- sidebar model and system stats
- browse and reload chat history from saved Q&A logs
- document upload and auto-ingest
- compact answer metadata and source filenames
=======
# simple_rag
Simple RAG chatbot

