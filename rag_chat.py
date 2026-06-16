"""
RAG Chat - CLI interface with automatic ingestion.

Run:  python rag_chat.py

Answers ONLY from retrieved document context.
On startup, automatically detects and ingests any new or modified PDFs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import requests
from chromadb.api.models.Collection import Collection

import config
from loaders import LOADERS, load_document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / "chat.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

QA_LOG = config.LOGS_DIR / "qa_history.jsonl"
_EMBED_URL = f"{config.OLLAMA_BASE_URL}/api/embed"
_EMBED_URL_LEGACY = f"{config.OLLAMA_BASE_URL}/api/embeddings"

# Some models (e.g. Qwen3) still emit an empty <think>...</think> block even
# with /no_think, and may echo the /no_think token itself. Strip both so the
# user never sees raw control tokens or empty reasoning blocks as a prefix.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NOTHINK_ECHO_RE = re.compile(r"^\s*/no_think\s*", re.IGNORECASE)
_SOURCE_REF_RE = re.compile(r"\[Source\s+(\d+)(?:[^\]]*)?\]")
_REASONING_PREFIX_RE = re.compile(
    r"^(?:\s*(?:step\s*\d+|to answer your question|to answer|from the provided documents)"
    r"[\s:.-]*)+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers: hashing / tracking / cache
# ---------------------------------------------------------------------------


def file_hash(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _read_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("%s is invalid JSON; starting fresh", path.name)
    return default


def load_tracking() -> dict:
    return _read_json(config.PROCESSED_FILES_PATH, {})


def save_tracking(tracking: dict) -> None:
    config.PROCESSED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_FILES_PATH.write_text(
        json.dumps(tracking, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_chunk_cache() -> dict:
    if config.CHUNKS_PKL_PATH.exists():
        try:
            with open(config.CHUNKS_PKL_PATH, "rb") as fh:
                data = pickle.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Could not load chunk cache: %s", exc)
    return {}


def save_chunk_cache(cache: dict) -> None:
    with open(config.CHUNKS_PKL_PATH, "wb") as fh:
        pickle.dump(cache, fh)


# ---------------------------------------------------------------------------
# Helpers: chunking / embedding / collection
# ---------------------------------------------------------------------------


def chunk_text(text: str) -> list[str]:
    """Character-level sliding-window chunker."""
    chunks, start = [], 0
    while start < len(text):
        end = start + config.CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - config.CHUNK_OVERLAP
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed via Ollama; falls back to legacy single-item endpoint if needed."""

    def _post(url: str, payload: dict) -> dict:
        resp = requests.post(url, json=payload, timeout=None)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"{resp.status_code} {resp.text}") from exc
        return resp.json()

    try:
        data = _post(_EMBED_URL, {"model": config.EMBEDDING_MODEL, "input": texts})
        if embeddings := data.get("embeddings"):
            return embeddings
        # Legacy fallback: one request per text
        results = []
        for text in texts:
            data = _post(
                _EMBED_URL_LEGACY, {"model": config.EMBEDDING_MODEL, "prompt": text}
            )
            if not (vec := data.get("embedding")):
                raise RuntimeError("Ollama embeddings API returned no vector.")
            results.append(vec)
        return results
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama. Run `ollama serve` and "
            f"`ollama pull {config.EMBEDDING_MODEL}`."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            f"Ollama embeddings failed for {config.EMBEDDING_MODEL}: {exc}"
        ) from exc


def get_or_create_collection(client: chromadb.PersistentClient) -> Collection:
    return client.get_or_create_collection(
        name=config.VECTOR_DB_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def delete_file_vectors(collection: Collection, filepath: str) -> None:
    collection.delete(where={"filepath": filepath})


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _purge_non_pdf(collection: Collection, tracking: dict, cache: dict) -> None:
    """Drop any previously indexed non-PDF entries in-place."""
    removed = [fp for fp in list(tracking) if Path(fp).suffix.lower() != ".pdf"]
    for fp in removed:
        delete_file_vectors(collection, fp)
        tracking.pop(fp, None)
        cache.pop(fp, None)
    if removed:
        logger.info(
            "Removed %d non-PDF record(s): %s",
            len(removed),
            [Path(f).name for f in removed[:20]],
        )
        save_tracking(tracking)
        save_chunk_cache(cache)


def auto_ingest_files(collection: Collection, force_reingest: bool = False) -> int:
    """Scan data/ and ingest new/modified PDFs. Returns number of files processed."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracking = load_tracking()
    cache = load_chunk_cache()
    _purge_non_pdf(collection, tracking, cache)

    supported = {ext.lstrip(".").lower() for ext in LOADERS}
    all_files = [p for p in config.DATA_DIR.rglob("*") if p.is_file()]
    files = [p for p in all_files if p.suffix.lstrip(".").lower() in supported]

    logger.info(
        "Scanning %d file(s); supported: %s",
        len(all_files),
        ", ".join(sorted(supported)),
    )

    skipped = [
        p.name for p in all_files if p.suffix.lstrip(".").lower() not in supported
    ]
    if skipped:
        logger.info("Skipped %d unsupported file(s): %s", len(skipped), skipped[:20])

    if not files:
        logger.warning("No supported documents found in %s", config.DATA_DIR)
        return 0

    if not force_reingest and collection.count() == 0:
        force_reingest = True
        logger.info("Vector DB empty; forcing full re-ingest")

    processed = 0
    for path in files:
        filepath = str(path.resolve())
        current_hash = file_hash(path)
        record = tracking.get(filepath, {})

        if not force_reingest and record.get("file_hash") == current_hash:
            logger.info("Skipping %s – unchanged", path.name)
            continue

        logger.info("Processing %s", path.name)

        try:
            text = load_document(path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)
            continue

        if not text.strip():
            logger.warning("No text extracted from %s – skipping", path.name)
            continue

        chunks = chunk_text(text)
        if not chunks:
            logger.warning("No chunks created for %s – skipping", path.name)
            continue

        try:
            embeddings = embed_texts(chunks)
        except Exception as exc:
            logger.error("Failed to embed %s: %s", path.name, exc)
            continue

        # Remove stale vectors before upserting
        if record.get("file_hash") or force_reingest:
            delete_file_vectors(collection, filepath)

        ids = [f"{current_hash}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "filepath": filepath,
                "filename": path.name,
                "file_hash": current_hash,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "file_ext": path.suffix.lower().lstrip("."),
            }
            for i in range(len(chunks))
        ]

        for start in range(0, len(chunks), 256):
            collection.upsert(
                ids=ids[start : start + 256],
                embeddings=embeddings[start : start + 256],
                documents=chunks[start : start + 256],
                metadatas=metadatas[start : start + 256],
            )

        entry = {
            "filename": path.name,
            "filepath": filepath,
            "file_hash": current_hash,
            "last_processed_time": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
        }
        tracking[filepath] = entry
        cache[filepath] = {**entry, "chunks": chunks, "metadatas": metadatas}
        save_tracking(tracking)
        save_chunk_cache(cache)
        processed += 1
        logger.info("Ingested: %s (%d chunks)", path.name, len(chunks))

    return processed


# ---------------------------------------------------------------------------
# Retrieval & generation
# ---------------------------------------------------------------------------


def retrieve(query: str, collection: Collection) -> list[dict]:
    query_vec = embed_texts([query])[0]
    response = collection.query(
        query_embeddings=[query_vec],
        n_results=config.TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if (score := round(1.0 - float(dist), 4)) >= config.SIMILARITY_THRESHOLD:
            chunks.append(
                {
                    "text": doc,
                    "filename": meta["filename"],
                    "filepath": meta["filepath"],
                    "chunk_index": meta["chunk_index"],
                    "file_hash": meta.get("file_hash", ""),
                    "chunk_count": meta.get("chunk_count", 0),
                    "score": score,
                }
            )
    return chunks


def _extract_used_source_indexes(answer: str) -> list[int]:
    indexes: list[int] = []
    for match in _SOURCE_REF_RE.finditer(answer or ""):
        idx = int(match.group(1))
        if idx not in indexes:
            indexes.append(idx)
    return indexes


def _strip_source_refs(answer: str) -> str:
    answer = _SOURCE_REF_RE.sub("", answer or "")
    lines = []
    for line in answer.splitlines():
        line = re.sub(r"\s+\.", ".", line)
        line = re.sub(r"\s+,", ",", line)
        lines.append(line.rstrip())
    answer = "\n".join(lines)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def _clean_answer_text(answer: str) -> str:
    answer = _THINK_BLOCK_RE.sub("", answer or "").strip()
    answer = _NOTHINK_ECHO_RE.sub("", answer).strip()
    answer = _REASONING_PREFIX_RE.sub("", answer).strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer


def _match_greeting(question: str) -> bool:
    """Return True if *question* is a simple greeting/small-talk message."""
    normalized = " ".join(question.strip().lower().split())
    if not normalized:
        return False
    greeting_prefixes = (
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "whats up",
        "what's up",
        "hello",
        "hey",
        "hi",
        "thanks",
        "thank you",
    )
    for phrase in greeting_prefixes:
        if normalized == phrase or normalized.startswith(f"{phrase} "):
            return True
        if phrase == "how are you" and phrase in normalized:
            return True
    return False


def is_general_greeting(question: str) -> bool:
    """Detect simple greetings and small-talk so we can answer them directly."""
    return _match_greeting(question)


def build_greeting_prompt(question: str) -> str:
    return f"""/no_think
You are a warm, friendly assistant.

The user message is a greeting or light small talk.

Respond politely and enthusiastically, and make the user feel welcome and comfortable asking questions.
- be warm, brief, and human
- do not mention documents unless the user brings them up
- do not mention prompts, rules, or hidden reasoning
- do not use a rigid template
- mirror the user's tone naturally
- keep the reply in the same level of energy as the user
- if the user is very brief, keep the reply very brief too

User message:
{question}

Reply:"""



def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i}: {c['filename']} (score={c['score']})]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"""
You are an AI assistant operating in a Retrieval-Augmented Generation (RAG) system.

PRIMARY OBJECTIVE

Your only source of truth for factual answers is the retrieved context below.
Do not use prior knowledge, memory, assumptions, or outside sources for factual content.

CORE RULES

1. Ground answers in retrieved context only.

    * Use the retrieved documents as the only factual source.
    * Do not invent facts, examples, dates, definitions, or explanations that are not supported by the retrieved context.
    * Do not use general knowledge or external sources.
    * If the retrieved context does not contain enough information, say exactly:
     "I don't know based on the available documents."

2. Be concise and faithful.

    * Rewrite information in a natural, easy-to-understand way.
    * You may explain the same document facts in simpler words.
    * Follow the user's requested output format whenever possible.
    * If the user asks for a report, table, description, comparison, summary, introduction, step-by-step answer, or any other structure, format the answer that way.
    * Use headings, bullets, tables, numbered steps, paragraphs, or mixed layouts as needed to match the request.
    * Only fill those sections with information that can be supported by the retrieved context.
    * If a requested section is not supported by the documents, say that it is not covered.

3. Answer the user's intent, not just the words.

    * Focus on what the user is trying to understand or accomplish.
    * User greetings should be answered politely and warmly.
    * Combine relevant information from multiple retrieved sources when appropriate.
    * Match the user's requested level of detail and requested format when the documents support it.
    * If the user asks for a detailed report, give a detailed report.
    * If the user asks for a table, return a table.
    * If the user asks for a description, return a clear description.
    * If the user asks for multiple sections, organize the response into those sections.

4. Handle missing information honestly.

   * If the answer cannot be found in the retrieved context, say so clearly.
   * Do not guess.
   * Do not fabricate details.
   * Do not partially answer with outside knowledge.

5. Reduce hallucinations.

    * Never invent names, numbers, dates, specifications, requirements, procedures, policies, or technical details.
    * If uncertain, acknowledge uncertainty.
    * Prefer saying "I don't know" over making assumptions.

6. Preserve important details.

   * Keep critical numbers, configurations, requirements, versions, limitations, and constraints exactly as stated in the retrieved content.

7. Response style.

    * Be concise for simple questions and detailed for complex questions.
    * Prioritize clarity, readability, and the user's requested format.
    * Use bullet points, tables, step-by-step explanations, or report-style sections when asked.
    * Complete answers are necessary, but do not add unsupported information just to be more comprehensive.
    * If the user asks multiple things in one question, answer every part.

8. Contradictory information.

    * If retrieved sources conflict:

      * Mention the conflict.
      * Explain the differing information.
      * Do not arbitrarily choose one unless evidence supports it.

9. Follow-up questions.

   * Use conversation history when available.
   * Maintain context across the conversation.
   * Ask clarifying questions when the user's request is ambiguous.

RESPONSE WORKFLOW

Step 1: Understand the user's intent.
Step 2: Search the retrieved context for relevant information only.
Step 3: Determine whether the answer is fully supported, partially supported, or unsupported.
Step 4: Generate a clear and helpful response.
Step 5: If information is missing, explicitly state the limitation.
Step 6: Do not add unsupported extra knowledge.
Step 7: Verify that no unsupported claims are presented as facts.

IMPORTANT

* Retrieved documents are the primary source of truth.
* Hallucination is not allowed.
* If the answer is not in the documents, say so.
* Keep the answer grounded in the retrieved context even when you are making it more readable.
* If helpful, you may mention source numbers, but do not force citations if they make the answer awkward or incomplete.

Use retrieved documents as the source of facts, and only reorganize, summarize, or restate what is already present there.

Retrieved Context:
{context}

Question:
{question}

Answer:"""


def call_llm(prompt: str) -> tuple[str, dict]:
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=None,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = _clean_answer_text(data["response"])
        if not answer:
            answer = "[ERROR] Model returned an empty response after stripping reasoning tags."
        eval_count = data.get("eval_count")
        eval_duration_ns = data.get("eval_duration")
        tokens_per_s = None
        if eval_count and eval_duration_ns:
            eval_seconds = float(eval_duration_ns) / 1_000_000_000
            if eval_seconds > 0:
                tokens_per_s = round(float(eval_count) / eval_seconds, 2)
        return answer, {
            "eval_count": eval_count,
            "tokens_per_s": tokens_per_s,
        }
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] Cannot connect to Ollama. "
            f"Run `ollama serve` then `ollama pull {config.LLM_MODEL}`",
            {},
        )
    except Exception as exc:
        return f"[ERROR] LLM call failed: {exc}", {}


def answer_question(
    question: str,
    collection: Collection,
    conversation_id: str | None = None,
) -> str:
    t0 = time.time()

    if is_general_greeting(question):
        answer, llm_meta = call_llm(build_greeting_prompt(question))
        logger.info("Handled greeting without retrieval: %s", question)
        _log_qa(
            question,
            [],
            answer,
            time.time() - t0,
            0.0,
            llm_meta,
            conversation_id,
        )
        return answer

    chunks = retrieve(question, collection)

    if not chunks:
        answer = "I don't know based on the available documents."
        logger.info("No relevant chunks for: %s", question)
        _log_qa(question, [], answer, time.time() - t0, 0.0, {}, conversation_id)
        return answer

    t_gen = time.time()
    raw_answer, llm_meta = call_llm(build_prompt(question, chunks))
    used_source_indexes = _extract_used_source_indexes(raw_answer)
    answer = _strip_source_refs(raw_answer)
    if not answer:
        answer = raw_answer.strip() or "I don't know based on the available documents."
    used_chunks = [chunks[i - 1] for i in used_source_indexes if 1 <= i <= len(chunks)]
    _log_qa(
        question,
        used_chunks,
        answer,
        time.time() - t0,
        time.time() - t_gen,
        llm_meta,
        conversation_id,
    )
    return answer


def _log_qa(
    question: str,
    chunks: list[dict],
    answer: str,
    total_elapsed: float,
    generation_elapsed: float,
    llm_meta: dict,
    conversation_id: str | None = None,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "question": question,
        "model": config.LLM_MODEL,
        "retrieved_docs": [
            {
                "filename": c["filename"],
                "score": c["score"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
        "answer": answer,
        "generation_time_s": round(generation_elapsed, 3),
        "response_time_s": round(total_elapsed, 3),
        "eval_count": llm_meta.get("eval_count"),
        "tokens_per_s": llm_meta.get("tokens_per_s"),
    }
    QA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QA_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
HELP_TEXT = """
Commands:
  /help    – show this message
  /sources – list ingested files
  /chunks  – list cached chunks
  /quit    – exit
"""


def list_sources() -> None:
    tracking = load_tracking()
    if not tracking:
        print("No files ingested yet.")
        return
    print(f"\n{'File':<40} {'Chunks':>7}  {'Last processed'}")
    print("-" * 75)
    for r in tracking.values():
        print(
            f"{r['filename']:<40} {r.get('chunk_count', '?'):>7}  {r['last_processed_time'][:19]}"
        )
    print()


def list_chunk_cache() -> None:
    cache = load_chunk_cache()
    if not cache:
        print("No chunk cache found.")
        return
    print(f"\n{'File':<40} {'Chunks':>7}  {'Hash'}")
    print("-" * 85)
    for r in cache.values():
        print(
            f"{r['filename']:<40} {r.get('chunk_count', '?'):>7}  {r.get('file_hash', '')[:12]}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n╔════════════════════════════════════╗")
    print("║            Local RAG Chat          ║")
    print("╚════════════════════════════════════╝")
    print(f"  Model : {config.LLM_MODEL}")
    print(f"  Embed : {config.EMBEDDING_MODEL} (local Ollama)")
    print(f"  Top-K : {config.TOP_K}   Threshold: {config.SIMILARITY_THRESHOLD}")
    print(HELP_TEXT)

    print("Checking Ollama embedding endpoint...", end=" ", flush=True)
    try:
        embed_texts(["ping"])
    except Exception as exc:
        print("failed.")
        raise RuntimeError(f"Unable to initialize Ollama embeddings: {exc}") from exc
    print("ready.")

    config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    collection = get_or_create_collection(
        chromadb.PersistentClient(path=config.CHROMA_PATH)
    )

    print("Scanning for new documents...")
    n = auto_ingest_files(collection)
    total = collection.count()
    if n:
        print(f" Ingested {n} new/modified file(s)")
    elif total == 0:
        print(" No documents found. Add files to data/")
    else:
        print(f" Up to date ({total} vectors ready)")
    print()

    COMMANDS = {
        "/help": lambda: print(HELP_TEXT),
        "/sources": list_sources,
        "/chunks": list_chunk_cache,
    }

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            break
        if raw.lower() in COMMANDS:
            COMMANDS[raw.lower()]()
            continue

        print("Assistant: ", end="", flush=True)
        print(answer_question(raw, collection))
        print()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)
