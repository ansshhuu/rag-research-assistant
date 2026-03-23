from dataclasses import dataclass, field
from src.loaders import Document


@dataclass
class Chunk:
    """
    A single piece of a Document after splitting.
    Carries its own metadata — inherits from parent Document
    and adds chunk-specific fields.
    """
    content: str
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# CORE CHUNKER
# ─────────────────────────────────────────────

def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Splits a Document into overlapping chunks of roughly chunk_size characters.

    Why overlap?
        If a sentence is split across two chunks, neither chunk alone has
        full context. Overlap ensures the boundary content appears in both
        the previous and next chunk — so retrieval never misses it.

    Why character-based (not token-based)?
        Simpler, no tokenizer dependency. Good enough for P1.
        Token-based chunking comes in P3 when we hook up the LLM.
    """
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

        # If we're not at the very end, try to break at a sentence boundary
        # rather than mid-word. Looks for ". ", "? ", "! " near the end.
        if end < len(text):
            boundary = _find_sentence_boundary(text, end)
            if boundary:
                end = boundary

        chunk_text = text[start:end]

        if chunk_text.strip():  # skip whitespace-only slices
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,          # inherit all parent metadata
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                    "chunk_start_char": start,
                    "chunk_end_char": end,
                }
            ))
            chunk_index += 1

        # Move forward by chunk_size minus overlap
        # This is what creates the sliding window effect
        start = end - chunk_overlap

    return chunks


def _find_sentence_boundary(text: str, near: int, search_window: int = 80) -> int | None:
    """
    Looks backwards from `near` for a sentence-ending punctuation followed
    by a space. Returns the position just after that punctuation, or None
    if no boundary found within the search window.

    Example:
        text = "...the model failed. It then retried..."
        near = 120
        → finds the ". " at position 110, returns 111
    """
    search_start = max(0, near - search_window)
    snippet = text[search_start:near]

    # Walk backwards through the snippet looking for sentence endings
    for i in range(len(snippet) - 1, -1, -1):
        if snippet[i] in ".!?" and i + 1 < len(snippet) and snippet[i + 1] == " ":
            return search_start + i + 1  # position right after the punctuation

    return None  # no boundary found — caller will use hard cut


# ─────────────────────────────────────────────
# BATCH CHUNKER — processes a list of Documents
# ─────────────────────────────────────────────

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Runs chunk_document() over every Document in a list.
    This is what ingest.py will call after loading.
    """
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