import pytest
from src.loaders import Document
from src.chunker import chunk_document, chunk_documents, Chunk


# ── helpers ──────────────────────────────────────────────

def _make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(
        content=text,
        metadata={"source": source, "type": "txt", "filename": source}
    )


SHORT_TEXT = "Hello world. This is a short document."
LONG_TEXT = (
    "Retrieval Augmented Generation is a technique that combines "
    "information retrieval with language generation. "
    "It was introduced to address the knowledge cutoff problem in LLMs. "
    "The system first retrieves relevant documents from a vector store. "
    "Then it passes those documents as context to the language model. "
    "This allows the model to answer questions about data it was never trained on. "
    "Vector databases store embeddings which are dense numerical representations. "
    "Cosine similarity is used to find the most relevant chunks. "
    "Chunk size and overlap are the two most important hyperparameters. "
    "Too small a chunk loses context. Too large a chunk dilutes relevance."
)


# ── basic structure tests ─────────────────────────────────

def test_returns_list_of_chunks():
    doc = _make_doc(LONG_TEXT)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunks_are_non_empty():
    doc = _make_doc(LONG_TEXT)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.content.strip() != ""


def test_short_text_produces_one_chunk():
    doc = _make_doc(SHORT_TEXT)
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert SHORT_TEXT.strip() in chunks[0].content


def test_long_text_produces_multiple_chunks():
    doc = _make_doc(LONG_TEXT)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


# ── metadata tests ────────────────────────────────────────

def test_chunks_inherit_parent_metadata():
    doc = _make_doc(LONG_TEXT, source="myfile.txt")
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["source"] == "myfile.txt"
        assert chunk.metadata["type"] == "txt"


def test_chunks_have_index():
    doc = _make_doc(LONG_TEXT)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunks_have_char_positions():
    doc = _make_doc(LONG_TEXT)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert "chunk_start_char" in chunk.metadata
        assert "chunk_end_char" in chunk.metadata


# ── overlap tests ─────────────────────────────────────────

def test_overlap_creates_shared_content():
    """
    With overlap=50, consecutive chunks should share some text.
    We check that the end of chunk[0] appears somewhere in chunk[1].
    """
    doc = _make_doc(LONG_TEXT)
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=50)
    if len(chunks) >= 2:
        tail_of_first = chunks[0].content[-30:]
        assert tail_of_first in chunks[1].content


def test_no_overlap_chunks_do_not_repeat():
    doc = _make_doc(LONG_TEXT)
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=0)
    if len(chunks) >= 2:
        tail_of_first = chunks[0].content[-30:]
        assert tail_of_first not in chunks[1].content


# ── edge case tests ───────────────────────────────────────

def test_empty_document_returns_empty_list():
    doc = _make_doc("")
    chunks = chunk_document(doc)
    assert chunks == []


def test_invalid_chunk_size_raises():
    doc = _make_doc(LONG_TEXT)
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(doc, chunk_size=0)


def test_overlap_larger_than_size_raises():
    doc = _make_doc(LONG_TEXT)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)


def test_negative_overlap_raises():
    doc = _make_doc(LONG_TEXT)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(doc, chunk_size=100, chunk_overlap=-1)


# ── batch chunker tests ───────────────────────────────────

def test_chunk_documents_processes_all():
    docs = [_make_doc(LONG_TEXT, f"doc{i}.txt") for i in range(3)]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 3  # must be more than one chunk per doc


def test_chunk_documents_empty_list():
    chunks = chunk_documents([], chunk_size=200, chunk_overlap=20)
    assert chunks == []