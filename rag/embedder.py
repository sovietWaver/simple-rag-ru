import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-m3"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model {_MODEL_NAME}...")
        _model = SentenceTransformer(_MODEL_NAME)
        print("Model loaded.")
    return _model


def chunk_to_vector(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


def chunks_to_vectors(texts: list[str], batch_size: int = 8) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vecs.astype(np.float32)
