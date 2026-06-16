from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

CHROMA_PATH = str(VECTOR_DB_DIR)
PROCESSED_FILES_PATH = VECTOR_DB_DIR / "processed_files.json"
CHUNKS_PKL_PATH = VECTOR_DB_DIR / "chunks.pkl"

VECTOR_DB_COLLECTION = "rag_docs"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

TOP_K = 5
SIMILARITY_THRESHOLD = 0.25

LLM_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = "24h"

EMBEDDING_MODEL = "bge-m3:latest"
EMBEDDING_DIM = 1024