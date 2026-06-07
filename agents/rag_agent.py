import chromadb
import ollama

from agents.state import PolicyState
from utils.model_config import get as get_model


_chroma_client = chromadb.Client()


def _collection_name(session_id: str) -> str:
    return f"policy_{session_id[:8]}"


def semantic_chunk(ocr_results: list, chunk_size: int = 400, overlap: int = 50) -> list:
    """Clause-aware chunking with metadata."""
    chunks = []
    chunk_id = 0

    for page_data in ocr_results:
        text = page_data.get("text", "")
        page = page_data.get("page", 0)
        section = page_data.get("section", "")

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) < chunk_size:
                buffer += " " + para
            else:
                if buffer.strip():
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id:04d}",
                        "text": buffer.strip(),
                        "metadata": {
                            "page": page,
                            "section": section,
                            "chunk_index": chunk_id,
                        },
                    })
                    chunk_id += 1
                buffer = para

        if buffer.strip():
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "text": buffer.strip(),
                "metadata": {
                    "page": page,
                    "section": section,
                    "chunk_index": chunk_id,
                },
            })
            chunk_id += 1

    return chunks


def embed_and_store_node(state: PolicyState) -> PolicyState:
    print("[RAG] Chunking policy document...")
    chunks = semantic_chunk(state["ocr_text"])
    embed_model = get_model("EMBED_MODEL", "nomic-embed-text")

    collection_name = _collection_name(state["session_id"])
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = _chroma_client.create_collection(collection_name)

    print(f"[RAG] Embedding {len(chunks)} chunks with {embed_model}...")
    for chunk in chunks:
        embedding = ollama.embeddings(
            model=embed_model,
            prompt=chunk["text"],
        )["embedding"]

        collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )

    # Build verbatim chunk index for the validator (chunk_id -> text + meta).
    chunk_index = {
        c["chunk_id"]: {
            "text": c["text"],
            "page": c["metadata"].get("page"),
            "section": c["metadata"].get("section", ""),
        }
        for c in chunks
    }

    print(f"[RAG] Stored {len(chunks)} chunks in ChromaDB.")
    return {**state,
            "chunks": chunks,
            "chunk_index": chunk_index,
            "status": "rag_complete",
            "active_node": "embed_store"}


def retrieve_chunks(session_id: str, query: str, n_results: int = 8) -> list:
    embed_model = get_model("EMBED_MODEL", "nomic-embed-text")
    collection = _chroma_client.get_collection(_collection_name(session_id))

    query_embedding = ollama.embeddings(
        model=embed_model,
        prompt=query,
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))
