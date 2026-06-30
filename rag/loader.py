import re
from pathlib import Path


def _clean(text: str) -> str:
    # Collapse runs of 3+ blank lines into two (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize non-breaking spaces and other Unicode space variants
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def load_file(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        text = p.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)

    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .txt, .pdf")

    return _clean(text)
