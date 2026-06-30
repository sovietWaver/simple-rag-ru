from langchain_text_splitters import RecursiveCharacterTextSplitter

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=_SEPARATORS,
    length_function=len,
    keep_separator=True,
)


def chunk_text(text: str, source: str) -> list[dict]:
    raw_chunks = _splitter.split_text(text)
    total = len(raw_chunks)
    return [
        {
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_index": i,
                "total_chunks": total,
            },
        }
        for i, chunk in enumerate(raw_chunks)
    ]
