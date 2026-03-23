import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    content: str
    metadata: dict


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _validate_path(path: Path, allowed_ext: list[str]):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in allowed_ext:
        raise ValueError(f"Expected {allowed_ext}, got: {path.suffix}")


# ─────────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    path = Path(file_path)
    _validate_path(path, [".pdf"])

    documents = []

    try:
        with fitz.open(file_path) as pdf:
            total_pages = len(pdf)

            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()

                if not text:
                    continue

                documents.append(Document(
                    content=text,
                    metadata={
                        "source": str(path.resolve()),
                        "type": "pdf",
                        "page": page_num,
                        "total_pages": total_pages,
                        "filename": path.name,
                    }
                ))

    except Exception as e:
        raise RuntimeError(f"Failed to read PDF '{file_path}': {e}")

    print(f"[PDF] Loaded {len(documents)} pages from '{path.name}'")
    return documents


# ─────────────────────────────────────────────
# WEB PAGE LOADER
# ─────────────────────────────────────────────

def load_webpage(url: str, timeout: int = 10) -> List[Document]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch URL '{url}': {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # remove noisy tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = soup.get_text(separator="\n", strip=True)

    if not text:
        raise ValueError(f"No text content found at URL: {url}")

    print(f"[WEB] Loaded '{title}'")

    return [Document(
        content=text,
        metadata={
            "source": url,
            "type": "webpage",
            "title": title,
        }
    )]


# ─────────────────────────────────────────────
# TEXT / MARKDOWN LOADER
# ─────────────────────────────────────────────

def load_text_file(file_path: str) -> List[Document]:
    path = Path(file_path)
    _validate_path(path, [".txt", ".md"])

    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read file '{file_path}': {e}")

    if not text:
        raise ValueError(f"File is empty: {file_path}")

    print(f"[TEXT] Loaded '{path.name}' ({len(text)} chars)")

    return [Document(
        content=text,
        metadata={
            "source": str(path.resolve()),
            "type": path.suffix.lstrip("."),
            "filename": path.name,
        }
    )]


# ─────────────────────────────────────────────
# UNIFIED LOADER
# ─────────────────────────────────────────────

def load_document(source: str) -> List[Document]:
    source = source.strip()

    if not source:
        raise ValueError("Source cannot be empty")

    if source.startswith(("http://", "https://")):
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