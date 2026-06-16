"""Standalone ingestion entry point for the PDF-only local RAG pipeline."""

from __future__ import annotations

import logging
import sys

import chromadb

import config
from rag_chat import auto_ingest_files, embed_texts, get_or_create_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / "ingest.log", encoding="utf-8"),
    ],
)


def main() -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("Checking Ollama embedding endpoint...", end=" ", flush=True)
    embed_texts(["ping"])
    print("ready.")

    config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    collection = get_or_create_collection(
        chromadb.PersistentClient(path=config.CHROMA_PATH)
    )

    print("Scanning for new documents...")
    processed = auto_ingest_files(collection)
    print(f"Done. Processed {processed} file(s).")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)
