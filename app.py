from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import chromadb
import streamlit as st

import config
from rag_chat import (
    auto_ingest_files,
    answer_question,
    get_or_create_collection,
    load_tracking,
)

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


APP_TITLE = "AnchorMind"
APP_TAGLINE = "Workspace: Local RAG"
SUPPORTED_UPLOADS = ["pdf", "docx", "txt", "md", "csv", "json", "pptx", "xlsx"]
QA_LOG = config.LOGS_DIR / "qa_history.jsonl"
CURRENT_CHAT_ID_KEY = "current_chat_id"


def load_css() -> None:
    css_path = Path("public/custom.css")
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner=False)
def get_collection():
    config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return get_or_create_collection(client)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("_bootstrapped", False)
    st.session_state.setdefault(CURRENT_CHAT_ID_KEY, uuid4().hex)
    st.session_state.setdefault("history_expanded", False)


def bootstrap_index() -> None:
    if st.session_state.get("_bootstrapped"):
        return
    auto_ingest_files(get_collection())
    st.session_state["_bootstrapped"] = True


def read_qa_history(limit: int = 50, search: str = "") -> list[dict]:
    if not QA_LOG.exists():
        return []

    records: list[dict] = []
    with open(QA_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if search and search.lower() not in record.get("question", "").lower():
                continue
            records.append(record)
    return records[-limit:]


def _thread_title(records: list[dict]) -> str:
    for record in records:
        question = " ".join(record.get("question", "").split())
        if question:
            return question
    return "Untitled chat"


def _thread_label(thread: dict) -> str:
    title = _thread_title(thread.get("records", []))
    timestamp = thread.get("last_timestamp", "") or thread.get("first_timestamp", "")
    time_label = "Unknown time"
    if timestamp:
        try:
            time_label = datetime.fromisoformat(timestamp).astimezone().strftime(
                "%b %d, %I:%M %p"
            )
        except Exception:
            time_label = timestamp[:16]

    label = title[:64]
    if len(title) > 64:
        label += "..."
    turns = thread.get("turn_count", 0)
    turn_label = "turn" if turns == 1 else "turns"
    return f"{time_label} - {label} ({turns} {turn_label})"


def read_chat_threads(search: str = "", limit: int = 25) -> list[dict]:
    records = read_qa_history(limit=5000)
    if not records:
        return []

    threads: dict[str, dict] = {}
    order: list[str] = []
    legacy_index = 0

    for record in records:
        conversation_id = record.get("conversation_id")
        if not conversation_id:
            conversation_id = f"legacy-{legacy_index}"
            legacy_index += 1

        thread = threads.get(conversation_id)
        if thread is None:
            thread = {
                "conversation_id": conversation_id,
                "records": [],
                "first_timestamp": record.get("timestamp", ""),
                "last_timestamp": record.get("timestamp", ""),
            }
            threads[conversation_id] = thread
            order.append(conversation_id)

        thread["records"].append(record)
        thread["last_timestamp"] = record.get("timestamp", thread["last_timestamp"])

    result = []
    search_lower = search.lower().strip()
    for conversation_id in order:
        thread = threads[conversation_id]
        if search_lower and not any(
            search_lower in (record.get("question", "") or "").lower()
            or search_lower in (record.get("answer", "") or "").lower()
            for record in thread["records"]
        ):
            continue
        thread["turn_count"] = len(thread["records"])
        result.append(thread)

    result.sort(key=lambda item: item.get("last_timestamp", ""), reverse=True)
    return result[:limit]


def latest_record(conversation_id: str | None = None, question: str = "") -> dict:
    records = read_qa_history(limit=5000)
    for record in reversed(records):
        if conversation_id and record.get("conversation_id") == conversation_id:
            return record
        if not conversation_id and record.get("question") == question:
            return record
    return records[-1] if records else {}


def get_index_overview(collection) -> dict:
    tracking = load_tracking()
    indexed_files = len(tracking)
    last_sync = ""
    if tracking:
        last_sync = max(
            (record.get("last_processed_time", "") for record in tracking.values()),
            default="",
        )
    return {
        "indexed_files": indexed_files,
        "chunks": collection.count(),
        "last_sync": last_sync,
    }


def _render_card(title: str, value: str, subtle: str = "") -> None:
    st.sidebar.markdown(
        f"""
        <div class="rag-card">
          <div class="rag-card-title">{title}</div>
          <div class="rag-card-value">{value}</div>
          <div class="rag-card-subtle">{subtle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_chat_button(thread: dict, index: int) -> bool:
    title = _thread_title(thread.get("records", []))
    label = title[:34] + ("..." if len(title) > 34 else "")
    return st.sidebar.button(
        label,
        key=f"recent_chat_{thread.get('conversation_id', index)}",
        use_container_width=True,
        type="secondary",
    )


def _render_sidebar_history() -> None:
    st.sidebar.markdown("### Search History")
    history_search = st.sidebar.text_input(
        "Search past questions",
        placeholder="Search by question text",
        key="history_search",
        label_visibility="collapsed",
    )

    history_threads = read_chat_threads(history_search, limit=25)
    if not history_threads:
        st.sidebar.caption("No saved chats yet. Ask a question to create one.")
        return

    st.sidebar.caption(f"{len(history_threads)} recent chat(s)")
    expanded = st.session_state.get("history_expanded", False)
    visible_threads = history_threads if expanded else history_threads[:4]

    for idx, thread in enumerate(visible_threads):
        if _render_recent_chat_button(thread, idx):
            load_history_chat(thread)
            st.rerun()

    if len(history_threads) > 4:
        button_label = "Show less" if expanded else "Show more"
        if st.sidebar.button(button_label, use_container_width=True, type="secondary"):
            st.session_state["history_expanded"] = not expanded
            st.rerun()


def render_assistant_footer(record: dict) -> None:
    total_time = record.get("response_time_s")
    model_name = record.get("model", config.LLM_MODEL)
    tokens_per_s = record.get("tokens_per_s")
    eval_count = record.get("eval_count")
    footer_parts = []
    if tokens_per_s is not None:
        footer_parts.append(f"Tokens/s {tokens_per_s:.1f}")
    if total_time is not None:
        footer_parts.append(f"Total {total_time:.1f}s")
    if eval_count is not None:
        footer_parts.append(f"{eval_count} tokens")
    footer_parts.append(model_name)
    st.caption(" | ".join(footer_parts))

    docs = record.get("retrieved_docs", [])
    names = []
    seen = set()
    for doc in docs:
        name = doc.get("filename", "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    if names:
        with st.expander("Sources", expanded=False):
            for name in names:
                st.markdown(f"- {name}")


def load_history_chat(thread: dict) -> None:
    messages = []
    for record in thread.get("records", []):
        question = record.get("question", "")
        answer = record.get("answer", "")
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append(
                {"role": "assistant", "content": answer, "metadata": record}
            )
    st.session_state["messages"] = messages
    st.session_state[CURRENT_CHAT_ID_KEY] = thread.get("conversation_id", uuid4().hex)


def render_sidebar(collection) -> None:
    overview = get_index_overview(collection)
    cpu = psutil.cpu_percent(interval=0.1) if psutil else None
    ram = psutil.virtual_memory().percent if psutil else None

    st.sidebar.markdown(
        """
        <div class="rag-sidebar-brand">
          <div class="rag-sidebar-logo">R</div>
          <div>
            <div class="rag-sidebar-title">AnchorMind</div>
            <div class="rag-sidebar-subtitle">Local RAG workspace</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("New Chat", use_container_width=True, type="primary"):
        st.session_state["messages"] = []
        st.session_state[CURRENT_CHAT_ID_KEY] = uuid4().hex
        st.rerun()

    st.sidebar.markdown("### Documents")
    _render_card("Indexed Files", str(overview["indexed_files"]), "Files in the vector database")
    if overview["last_sync"]:
        try:
            sync_label = datetime.fromisoformat(overview["last_sync"]).astimezone().strftime(
                "%b %d, %I:%M %p"
            )
        except Exception:
            sync_label = overview["last_sync"][:19]
    else:
        sync_label = "Never"
    _render_card("Last Sync", sync_label, "Most recent document index update")

    st.sidebar.markdown("### System")
    _render_card("LLM", config.LLM_MODEL, "Primary answer model")
    _render_card("Embedding", config.EMBEDDING_MODEL, "Vector model for retrieval")
    _render_card("Chunks", str(overview["chunks"]), "Stored embeddings in ChromaDB")
    _render_card("Vector DB", "Connected", "Persistent ChromaDB index")

    st.sidebar.markdown("### Resources")
    if cpu is not None and ram is not None:
        for label, value in (("CPU", int(cpu)), ("RAM", int(ram))):
            st.sidebar.markdown(
                f"""
                <div class="rag-resource-row">
                  <span class="rag-resource-label">{label}</span>
                  <span class="rag-resource-value">{value}%</span>
                </div>
                <div class="rag-meter-track">
                  <div class="rag-meter-fill" style="width: {value}%"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("Resource monitoring unavailable on this machine.")

    st.sidebar.markdown("### Chat Actions")
    if st.sidebar.button("Refresh", use_container_width=True, type="primary"):
        st.rerun()

    _render_sidebar_history()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    init_state()

    collection = get_collection()
    bootstrap_index()
    render_sidebar(collection)

    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                "**Hello.** Ask anything about your documents and I will answer in a clear, structured way."
            )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("metadata"):
                render_assistant_footer(message["metadata"])

    prompt = st.chat_input("Ask anything about your documents...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = answer_question(
                        prompt,
                        collection,
                        conversation_id=st.session_state[CURRENT_CHAT_ID_KEY],
                    )
                record = latest_record(
                    conversation_id=st.session_state[CURRENT_CHAT_ID_KEY],
                    question=prompt,
                )
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "metadata": record,
                    }
                )
                st.rerun()
            except Exception as exc:
                error_text = f"Error: {exc}"
                st.error(error_text)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": error_text,
                        "metadata": {},
                    }
                )


if __name__ == "__main__":
    main()
