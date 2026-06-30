import chromadb
from rag.embedder import chunk_to_vector


def search(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    query_vec = chunk_to_vector(query)
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": text, "metadata": meta, "distance": dist})
    return hits


def build_context(hits: list[dict]) -> str:
    parts = []
    for hit in hits:
        source = hit["metadata"].get("source", "unknown")
        parts.append(f"[Источник: {source}]\n{hit['text']}")
    return "\n\n---\n\n".join(parts)
