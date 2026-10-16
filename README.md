<p align="center">
  <img src="docs/assets/banner.svg" alt="DocChat: upload your docs, get answers with sources" width="100%">
</p>

<h3 align="center">Built by Emrecan Gök</h3>

<p align="center">
  <a href="https://github.com/Emrecangok"><img src="https://img.shields.io/badge/GitHub-Emrecangok-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://www.linkedin.com/in/emrecan-g%C3%B6k-864b80248/"><img src="https://img.shields.io/badge/LinkedIn-Emrecan%20G%C3%B6k-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/LangGraph-agentic%20RAG-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/OpenAI-gpt--6--luna-412991?logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Chroma-vector%20store-00854A" alt="Chroma">
  <img src="https://img.shields.io/badge/Docling-PDF%20%C2%B7%20DOCX%20%C2%B7%20XLSX-5B6B63" alt="Docling">
</p>

**DocChat is a chatbot for your own documents.** Upload PDF, Word, PowerPoint, Excel, Markdown or HTML files and ask questions in plain language. An agent decides where to search, checks that what it found actually answers the question, and replies with the file and page it came from. If the answer isn't in your files, it says so instead of guessing.

<p align="center">
  <img src="docs/assets/stack.svg" alt="Tech stack: GPT API, LangChain, LangGraph, RAG, Docling, Chroma, Streamlit" width="100%">
</p>

<!-- Demo: add docs/assets/demo.gif, then remove this comment line and the one below
<p align="center"><img src="docs/assets/demo.gif" alt="Uploading a PDF and asking a question in DocChat" width="85%"></p>
-->

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="images/2.png" alt="DocChat answering with cited sources and saying when information is not in the documents"></td>
    <td width="50%"><img src="images/1.png" alt="DocChat listing the uploaded documents"></td>
  </tr>
  <tr>
    <td align="center"><sub>Answers with cited sources, and an honest "not found"</sub></td>
    <td align="center"><sub>Knows which documents are uploaded</sub></td>
  </tr>
</table>

## Features

- Upload files and ask a question in the same message
- Answers come only from the uploaded documents, with sources (`file · page`)
- The agent picks which document to search, or asks you when it can't tell
- Retrieved passages are graded for relevance; the query is rewritten if nothing fits
- Says "not found in your documents" instead of guessing
- Keeps the conversation short (last 6 messages) to control cost

## How it works

<p align="center">
  <img src="docs/assets/flow.svg" alt="DocChat agent flow: START, ingest, agent, run_tools, grade, then generate, rewrite or fallback, trim, END" width="100%">
</p>

| Node | What it does |
|---|---|
| `ingest` | Reads new files with Docling, splits them into chunks, stores them in Chroma |
| `agent` | LLM with two tools: `search_document` and `list_documents` |
| `run_tools` | LangGraph `ToolNode`; runs the tool the agent asked for |
| `grade_docs` | One LLM call decides if the retrieved chunks answer the question |
| `rewrite` | Writes a better search query and searches again (max 2 times) |
| `generate` | Writes the answer from the graded chunks, with citations |
| `fallback` | Used when nothing relevant is found after the retries |
| `trim` | When the chat reaches 12 messages, keeps the last 6 |

## Project structure

```
app.py        Streamlit chat UI
graph.py      LangGraph graph (nodes and edges)
nodes.py      Node functions and tools
state.py      Graph state
rag.py        Docling loading, chunking, Chroma vector store
llm.py        OpenAI chat model
config.py     Settings read from .env
prompts/      System, grade, rewrite and generate prompts
documents/    Uploaded files
chroma_db/    Vector store (created automatically)
```

## Setup

Python 3.11+ is required.

```bash
pip install streamlit langgraph langchain langchain-core langchain-openai langchain-chroma langchain-docling docling tiktoken python-dotenv
```

Create a `.env` file in the project folder:

```
OPENAI_API_KEY = sk-...
OPENAI_MODEL_NAME = gpt-6-luna
OPENAI_EMBEDDING_MODEL = text-embedding-3-small
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_RETRIES = 2

RAG_RETRIVER_K = 4
RAG_MAX_QUERY_RETRIES = 2
TRIM_TRIGGER = 12
TRIM_KEEP = 6
```

## Run

```bash
streamlit run app.py
```

Click **+** in the chat box to upload files, then ask a question. The first upload is slower because Docling loads its models.

## Notes

- If you change the embedding model, delete the `chroma_db/` folder and upload the files again. Vectors from different models can't be mixed.
- OCR is turned off for speed, so scanned PDFs (images of pages) won't produce text.
- `main.py` and `routers/` are a FastAPI skeleton that is not wired up yet.
