import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "chroma_db")
)

_retriever = None


def get_retriever():
    """Cached singleton — used by both the tool and the upfront retrieve_node."""
    global _retriever
    if _retriever is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": device},
        )
        store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        _retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 12, "lambda_mult": 0.7},
        )
    return _retriever


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("drug_name") or doc.metadata.get("source") or "unknown"
        chunks.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


@tool
def retrieve_drug_info(query: str) -> str:
    """Semantic search over the local FDA drug-label vector store.

    Use this for general medication questions, indications, contraindications,
    warnings, or anything that needs free-text context from the FDA labels.

    Args:
        query: a natural-language question or keyphrase about a drug or condition.

    Returns:
        Concatenated most-relevant FDA-label passages (with their source
        drug name) or a not-found message.
    """
    try:
        retriever = get_retriever()
    except Exception as exc:
        return f"[retrieve_drug_info error] Vector store unavailable: {exc}"

    docs = retriever.invoke(query)
    if not docs:
        return "No matching FDA documentation found in the local corpus."
    return format_docs(docs)
