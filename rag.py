import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

COLLECTION_NAME = "dosyalar"
PERSIST_DIR = "./chroma_db"
CHUNK_MAX_TOKENS = 512

embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


# ------------------------------------------------------------------
# Docling: ağır nesneler bir kere ve ilk ihtiyaçta oluşturulur
# ------------------------------------------------------------------
_converter = None
_chunker = None


def list_sources(vector_store:Chroma) -> list[str]:
    metadatas = vector_store.get(include=["metadatas"])["metadatas"]

    return sorted({m["source"] for m in metadatas if m and m.get("source")})

def document_overview(vector_store : Chroma, max_topics : int = 8) -> dict[str,list[str]]:
    metadatas = vector_store.get(include=["metadatas"])["metadatas"]

    overview : dict[str,list[str]] = {}

    for meta in metadatas:
        if not meta or not meta.get("source"):
            continue
        topics = overview.setdefault(meta["source"],[])


        heading = (meta.get("headings") or "").strip()

        if heading and heading not in topics and len(topics) < max_topics:
            topics.append(heading)

    return dict(sorted(overview.items()))

def format_document_list(overview: dict[str,list[str]]) -> str:
    if not overview:
        return "(no documents uploaded yet)"
    return "\n".join(
        f"- {name}: {','.join(topics) or'no headings'} " for name,topics in overview.items()
    )



def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
    return _converter


def _get_chunker():
    global _chunker
    if _chunker is None:
        import tiktoken
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.get_encoding("cl100k_base"),   # text-embedding-3-* tokenizer'ı
            max_tokens=CHUNK_MAX_TOKENS,
        )
        _chunker = HybridChunker(tokenizer=tokenizer)
    return _chunker


def _load_docling_chunks(file_path: str):
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS,
        converter=_get_converter(),
        chunker=_get_chunker(),
    )
    return loader.load()


# ------------------------------------------------------------------
# Vector store
# ------------------------------------------------------------------
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def load_chunks(docs: list[Document], batch_size: int = 100):
    vector_store = get_vector_store()
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        print(f"[embedding] {i + 1}-{i + len(batch)} / {len(docs)}", flush=True)
        vector_store.add_documents(batch)


def is_indexed(file_name: str) -> bool:
    return bool(get_vector_store().get(where={"source": file_name}, limit=1)["ids"])


def index_file(file_path: str) -> int:

    name = Path(file_path).name

    if is_indexed(name):
        print(f"{name} zaten indeksli, atlandı.")
        return 0

    start = time.time()
    docs = []
    for doc in _load_docling_chunks(file_path):
        meta = doc.metadata.get("dl_meta") or {}
        headings = meta.get("headings") or []

        page_numbers = []

        for item in meta.get("doc_items",[]):
            for prov in item.get("prov",[]):
                page_numbers.append(prov["page_no"])

        first_page = min(page_numbers) if page_numbers else -1

        docs.append(Document(
            page_content=doc.page_content,                 
            metadata={
                "source": name,
                "headings": " > ".join(headings),
                "page": first_page,          
            },
        ))
    print(f"{name}: {len(docs)} chunk, dönüştürme + chunking {time.time() - start:.1f} sn")

    load_chunks(docs)
    return len(docs)


def index_folder(folder: str = "documents") -> int:
    total = 0
    for path in sorted(Path(folder).glob("*.pdf")):
        total += index_file(str(path))
    return total



def docling_load_example(file_path: str = "documents/havacilik.pdf"):
    start = time.time()
    docs = _load_docling_chunks(file_path)
    print(f"Toplam chunk sayısı: {len(docs)}  ({time.time() - start:.1f} sn)")
    print("=" * 100)

    for i, doc in enumerate(docs):
        headings = (doc.metadata.get("dl_meta") or {}).get("headings") or ["Başlıksız"]
        print(f"\n--- Chunk {i} | Başlık: {' > '.join(headings)} ---")
        print(doc.page_content)
        print("-" * 50)


if __name__ == "__main__":
    docling_load_example()