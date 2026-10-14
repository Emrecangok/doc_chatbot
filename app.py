import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from graph import build_graph
from nodes import _extract_text
from rag import index_file, get_vector_store


UPLOAD_DIR = Path("documents")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = ["pdf", "docx", "pptx", "xlsx", "md", "html"]

WELCOME = (
    "Hi there! 👋 I'm **DocChat**. Upload your documents with the **+** button below, "
    "then ask me anything about them. I'll answer based on your files and show you the sources."
)


st.set_page_config(page_title="DocChat", page_icon="📄", layout="centered")

st.markdown("""
<style>
    .stApp { background: #F3F8F5; }
    .block-container { max-width: 760px; padding-top: 2rem; padding-bottom: 2rem; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Frame */
    .st-key-chat_frame {
        background: #FFFFFF;
        border: 2px solid #00854A !important;
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(0, 133, 74, 0.12);
    }
    .st-key-messages {
        border: none !important;
        background: #FFFFFF;
    }

    .bot-header {
        background: #00854A;
        color: #FFFFFF;
        padding: 18px 22px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0;
    }
    .bot-avatar {
        width: 44px; height: 44px;
        border-radius: 50%;
        background: #FFFFFF;
        color: #00854A;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
    }
    .bot-title { font-size: 18px; font-weight: 600; margin: 0; }
    .bot-status { font-size: 13px; opacity: .9; margin: 0; }
    .bot-status::before {
        content: ""; display: inline-block;
        width: 8px; height: 8px; border-radius: 50%;
        background: #7CFFB2; margin-right: 6px;
    }

    [data-testid="stChatMessage"] { border-radius: 12px; }

    .file-chip {
        display: inline-block;
        background: #EEF7F1;
        color: #00854A;
        border: 1px solid #CDE8D7;
        border-radius: 999px;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        font-size: 12px;
    }

    .sources-title {
        font-size: 12px;
        font-weight: 600;
        color: #5B6B63;
        margin: 10px 0 4px 0;
    }
    .source-chip {
        display: inline-block;
        background: #FFFFFF;
        color: #1F2D27;
        border: 1px solid #00854A;
        border-radius: 8px;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


@st.cache_resource
def get_graph():
    # built once per server, not on every rerun
    return build_graph(get_vector_store())


def format_sources(docs) -> list[str]:
    sources = []
    for d in docs:
        s = d.metadata.get("source", "?")
        if d.metadata.get("page", -1) > 0:
            s += f" · p.{d.metadata['page']}"
        if s not in sources:
            sources.append(s)
    return sources


def generate_answer(question: str, file_paths: list[str]) -> tuple[str, list[str]]:
    result = get_graph().invoke(
        {
            "messages": [HumanMessage(content=question)],
            "files": file_paths,
            # MemorySaver keeps these from the previous turn, so reset them
            "documents": [],
            "doc_grade": "",
            "retry_count": 0,
        },
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    answer = _extract_text(result["messages"][-1].content)
    return answer, format_sources(result.get("documents", []))


def render_sources(sources: list[str]):
    if not sources:
        return
    chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
    st.markdown(f'<p class="sources-title">Sources</p>{chips}', unsafe_allow_html=True)


def save_files(files) -> list[str]:
    names = []
    for file in files:
        (UPLOAD_DIR / file.name).write_bytes(file.getbuffer())
        names.append(file.name)
    return names


def index_files(names: list[str]):
    indexed = st.session_state.setdefault("indexed", set())
    for name in names:
        if name in indexed:
            continue  # don't embed the same file twice
        n = index_file(str(UPLOAD_DIR / name))
        indexed.add(name)
        st.toast(f"{name}: {n} chunks indexed")


def new_file_paths(names: list[str]) -> list[str]:
    # files the graph's ingest node should index (skip ones already in Chroma)
    indexed = st.session_state.setdefault("indexed", set())
    paths = [str(UPLOAD_DIR / n) for n in names if n not in indexed]
    indexed.update(names)
    return paths


with st.container(key="chat_frame"):

    # Header bar
    col_header, col_btn = st.columns([6, 1], vertical_alignment="center")
    with col_header:
        st.markdown("""
        <div class="bot-header">
            <div class="bot-avatar">📄</div>
            <div>
                <p class="bot-title">DocChat</p>
                <p class="bot-status">Answers from your documents, with sources</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        if st.button("🗑️", help="Clear chat"):
            st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
            st.session_state.thread_id = str(uuid.uuid4())  # graph forgets the old chat too
            st.rerun()

    # Messages (scrollable area)
    messages = st.container(height=430, key="messages")
    with messages:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📄"):
                if msg.get("files"):
                    st.markdown("".join(f'<span class="file-chip">📎 {f}</span>' for f in msg["files"]),
                                unsafe_allow_html=True)
                if msg["content"]:
                    st.markdown(msg["content"])
                render_sources(msg.get("sources", []))

    # Input: text + file upload in the same box, inside the frame
    user_input = st.chat_input(
        "Ask about your documents or click + to upload...",
        accept_file="multiple",
        file_type=ALLOWED_EXTENSIONS,
    )

if user_input:
    question = (user_input.text or "").strip()
    uploaded = save_files(user_input.files) if user_input.files else []

    if uploaded and not question:
        # files only: no question for the graph, index right here
        with st.spinner("Reading documents..."):
            index_files(uploaded)

    st.session_state.messages.append({"role": "user", "content": question, "files": uploaded})
    with messages:
        with st.chat_message("user", avatar="🧑"):
            if uploaded:
                st.markdown("".join(f'<span class="file-chip">📎 {f}</span>' for f in uploaded),
                            unsafe_allow_html=True)
            if question:
                st.markdown(question)

        with st.chat_message("assistant", avatar="📄"):
            with st.spinner("Searching your documents..."):
                if question:
                    # files + question: the graph's ingest node indexes them first
                    answer, sources = generate_answer(question, new_file_paths(uploaded))
                else:
                    answer = f"Received {len(uploaded)} document(s) ✅ You can now ask questions about them."
                    sources = []
            st.markdown(answer)
            render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
