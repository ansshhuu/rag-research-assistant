import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Document:
    """
    A single loaded document with its text content and metadata.
    Every loader returns a list of these — consistent shape across all sources.
    """
    content: str
    metadata: dict  # source, type, page (if PDF), title (if web)


# ─────────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> list[Document]:
    """
    Reads a PDF file page by page.
    Each page becomes its own Document so we can track page numbers in metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    documents = []
    pdf = fitz.open(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text().strip()

        if not text:          # skip blank/image-only pages
            continue

        documents.append(Document(
            content=text,
            metadata={
                "source": str(path.resolve()),
                "type": "pdf",
                "page": page_num + 1,   # 1-indexed, more readable
                "total_pages": len(pdf),
                "filename": path.name,
            }
        ))

    pdf.close()
    print(f"[PDF] Loaded {len(documents)} pages from '{path.name}'")
    return documents


# ─────────────────────────────────────────────
# WEB PAGE LOADER
# ─────────────────────────────────────────────

def load_webpage(url: str, timeout: int = 10) -> list[Document]:
    """
    Fetches a URL, strips HTML tags, returns clean text as a single Document.
    Raises clearly if the request fails so the pipeline doesn't silently skip.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()     # raises on 4xx / 5xx
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch URL '{url}': {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise: scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url
    text = soup.get_text(separator="\n", strip=True)

    if not text:
        raise ValueError(f"No text content found at URL: {url}")

    print(f"[WEB] Loaded '{title}' from '{url}'")
    return [Document(
        content=text,
        metadata={
            "source": url,
            "type": "webpage",
            "title": title,
        }
    )]


# ─────────────────────────────────────────────
# PLAIN TEXT / MARKDOWN LOADER
# ─────────────────────────────────────────────

def load_text_file(file_path: str) -> list[Document]:
    """
    Reads .txt or .md files as a single Document.
    Simple, but important — research notes, markdown docs, README files.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() not in [".txt", ".md"]:
        raise ValueError(f"Expected .txt or .md, got: {path.suffix}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"File is empty: {file_path}")

    print(f"[TEXT] Loaded '{path.name}' ({len(text)} chars)")
    return [Document(
        content=text,
        metadata={
            "source": str(path.resolve()),
            "type": path.suffix.lstrip("."),   # "txt" or "md"
            "filename": path.name,
        }
    )]


# ─────────────────────────────────────────────
# UNIFIED LOADER — the one function ingest.py will call
# ─────────────────────────────────────────────

def load_document(source: str) -> list[Document]:
    """
    Auto-detects source type and routes to the correct loader.
    - Starts with http/https → webpage
    - Ends with .pdf → PDF
    - Ends with .txt or .md → text file
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_webpage(source)

    path = Path(source)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return load_pdf(source)
    elif ext in [".txt", ".md"]:
        return load_text_file(source)
    else:
        raise ValueError(
            f"Unsupported source type '{ext}'. "
            f"Supported: .pdf, .txt, .md, or a URL."
        )