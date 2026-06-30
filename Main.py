import argparse
from pathlib import Path

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import chunks_to_vectors
from rag.store import get_client, get_collection, add_documents, reset_collection
from rag.searcher import search, build_context
from rag.generator import ask_llm_stream


def cmd_index(paths: list[str]) -> None:
    client = get_client()
    collection = get_collection(client)

    for raw_path in paths:
        p = Path(raw_path)
        if p.is_dir():
            files = list(p.glob("**/*.pdf")) + list(p.glob("**/*.txt"))
            if not files:
                print(f"No .pdf/.txt files found in {p}")
                continue
        elif p.is_file():
            files = [p]
        else:
            print(f"Path not found: {p}")
            continue

        for fpath in files:
            print(f"\nIndexing {fpath.name} ...")
            text = load_file(str(fpath))
            print(f"  Text length: {len(text)} chars")

            chunks = chunk_text(text, source=fpath.name)
            print(f"  Chunks: {len(chunks)}")

            texts = [c["text"] for c in chunks]
            embeddings = chunks_to_vectors(texts)

            add_documents(collection, chunks, embeddings)
            print(f"  Done: {fpath.name}")


def cmd_ask(question: str, top_k: int = 3) -> None:
    client = get_client()
    collection = get_collection(client)

    count = collection.count()
    if count == 0:
        print("Collection is empty. Run: python Main.py index <file>")
        return

    hits = search(collection, question, top_k=top_k)
    if not hits:
        print("No relevant chunks found.")
        return

    context = build_context(hits)
    print("\n--- Answer ---\n")
    ask_llm_stream(context, question)

    print("\n--- Sources ---")
    for hit in hits:
        m = hit["metadata"]
        dist = hit["distance"]
        print(f"  {m['source']}  chunk {m['chunk_index']}/{m['total_chunks'] - 1}  dist={dist:.4f}")


def cmd_reset() -> None:
    client = get_client()
    reset_collection(client)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Main",
        description="Local RAG pipeline for Russian documents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index file(s) or a directory")
    p_index.add_argument("paths", nargs="+", metavar="PATH")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", metavar="QUESTION")
    p_ask.add_argument("--top-k", type=int, default=3, dest="top_k")

    # reset
    sub.add_parser("reset", help="Delete and recreate the collection")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.paths)
    elif args.command == "ask":
        cmd_ask(args.question, top_k=args.top_k)
    elif args.command == "reset":
        cmd_reset()


if __name__ == "__main__":
    main()
