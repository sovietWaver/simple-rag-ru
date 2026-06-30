import numpy as np
import chromadb

COLLECTION_NAME = "rag_docs"
_CHROMA_PATH = "./chroma_db"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=_CHROMA_PATH)


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # cosine distance works correctly with L2-normalised embeddings
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    collection: chromadb.Collection,
    documents: list[dict],
    embeddings: np.ndarray,
    batch_size: int = 1000,
) -> None:
    ids = [
        f"{doc['metadata']['source']}::{doc['metadata']['chunk_index']}"
        for doc in documents
    ]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    emb_list = embeddings.tolist()

    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=emb_list[start:end],
        )
        print(f"  Stored chunks {start}–{end - 1} / {total - 1}")


def reset_collection(client: chromadb.PersistentClient, name: str = COLLECTION_NAME) -> None:
    try:
        client.delete_collection(name)
        print(f"Collection '{name}' deleted.")
    except Exception:
        print(f"Collection '{name}' did not exist, nothing to delete.")
    client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{name}' recreated.")
