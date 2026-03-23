from dataclasses import dataclass, field
from src.loaders import Document


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    text = document.content.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = _find_sentence_boundary(text, end)
            if boundary:
                end = boundary

        chunk_text = text[start:end]

        if chunk_text.strip():  
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,         
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                    "chunk_start_char": start,
                    "chunk_end_char": end,
                }
            ))
            chunk_index += 1
        start = end - chunk_overlap

    return chunks


def _find_sentence_boundary(text: str, near: int, search_window: int = 80) -> int | None:
    search_start = max(0, near - search_window)
    snippet = text[search_start:near]

    for i in range(len(snippet) - 1, -1, -1):
        if snippet[i] in ".!?" and i + 1 < len(snippet) and snippet[i + 1] == " ":
            return search_start + i + 1 

    return None  

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    print(
        f"[CHUNKER] {len(documents)} document(s) → "
        f"{len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
