import pytest
import numpy as np
from src.vector_store import VectorStore, _make_id, _sanitize_metadata


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """
    Creates a fresh VectorStore in a temp directory for each test.
    tmp_path is a built-in pytest fixture — gives a unique temp folder
    per test so tests never interfere with each other.
    """
    vs = VectorStore(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_collection",
    )
    yield vs
    vs.delete_collection()


def _make_embedded(content: str, dim: int = 384, index: int = 0, source: str = "test.txt"):
    """Helper — makes a fake embedded chunk with a random unit vector."""
    vec = np.random.rand(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)   # normalize to unit length
    return {
        "content":   content,
        "embedding": vec,
        "metadata":  {
            "source":      source,
            "type":        "txt",
            "chunk_index": index,
            "filename":    source,
        }
    }


# ── initialization tests ──────────────────────────────────

def test_store_initializes_empty(store):
    assert store.count() == 0


def test_store_collection_name(store):
    assert store.collection_name == "test_collection"


# ── add / count tests ─────────────────────────────────────

def test_add_single_chunk(store):
    chunks = [_make_embedded("Hello world.", index=0)]
    added = store.add(chunks)
    assert added == 1
    assert store.count() == 1


def test_add_multiple_chunks(store):
    chunks = [_make_embedded(f"Chunk {i}.", index=i) for i in range(5)]
    added = store.add(chunks)
    assert added == 5
    assert store.count() == 5


def test_add_empty_list(store):
    added = store.add([])
    assert added == 0
    assert store.count() == 0


def test_upsert_does_not_duplicate(store):
    """
    Adding the same chunk twice should update, not duplicate.
    Count should stay at 1, not go to 2.
    """
    chunk = [_make_embedded("Same content.", index=0, source="doc.txt")]
    store.add(chunk)
    store.add(chunk)    # second add — same source + index = same ID
    assert store.count() == 1


# ── query tests ───────────────────────────────────────────

def test_query_returns_results(store):
    chunks = [_make_embedded(f"Document about topic {i}.", index=i) for i in range(5)]
    store.add(chunks)

    query_vec = np.random.rand(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = store.query(query_vec, top_k=3)
    assert len(results) == 3


def test_query_result_structure(store):
    store.add([_make_embedded("Some content about RAG.", index=0)])

    query_vec = np.random.rand(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = store.query(query_vec, top_k=1)
    assert "content"  in results[0]
    assert "metadata" in results[0]
    assert "score"    in results[0]


def test_query_score_in_valid_range(store):
    chunks = [_make_embedded(f"Chunk {i}.", index=i) for i in range(3)]
    store.add(chunks)

    query_vec = np.random.rand(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = store.query(query_vec, top_k=3)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0


def test_query_top_k_respected(store):
    chunks = [_make_embedded(f"Item {i}.", index=i) for i in range(10)]
    store.add(chunks)

    query_vec = np.random.rand(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = store.query(query_vec, top_k=4)
    assert len(results) == 4


def test_query_empty_store_returns_empty(store):
    query_vec = np.random.rand(384).astype(np.float32)
    results = store.query(query_vec, top_k=5)
    assert results == []


def test_query_invalid_top_k_raises(store):
    query_vec = np.random.rand(384).astype(np.float32)
    with pytest.raises(ValueError, match="top_k"):
        store.query(query_vec, top_k=0)


def test_query_metadata_filter(store):
    """
    Metadata filtering lets you query only PDFs, or only a specific source.
    This is what makes RAG useful in multi-document setups.
    """
    pdf_chunk = _make_embedded("PDF content.", index=0, source="paper.pdf")
    pdf_chunk["metadata"]["type"] = "pdf"

    web_chunk = _make_embedded("Web content.", index=0, source="https://example.com")
    web_chunk["metadata"]["type"] = "webpage"

    store.add([pdf_chunk, web_chunk])

    query_vec = np.random.rand(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    pdf_results = store.query(query_vec, top_k=5, where={"type": "pdf"})
    assert all(r["metadata"]["type"] == "pdf" for r in pdf_results)


# ── utility tests ─────────────────────────────────────────

def test_get_all_sources_empty(store):
    assert store.get_all_sources() == []


def test_get_all_sources_returns_unique(store):
    chunks = [
        _make_embedded("Chunk 0.", index=0, source="doc_a.txt"),
        _make_embedded("Chunk 1.", index=1, source="doc_a.txt"),
        _make_embedded("Chunk 2.", index=0, source="doc_b.txt"),
    ]
    store.add(chunks)
    sources = store.get_all_sources()
    assert len(sources) == 2
    assert "doc_a.txt" in sources
    assert "doc_b.txt" in sources


# ── helper function tests ─────────────────────────────────

def test_make_id_is_deterministic():
    id1 = _make_id("myfile.txt", 3)
    id2 = _make_id("myfile.txt", 3)
    assert id1 == id2


def test_make_id_different_inputs_differ():
    id1 = _make_id("myfile.txt", 0)
    id2 = _make_id("myfile.txt", 1)
    assert id1 != id2


def test_sanitize_metadata_removes_none():
    meta = {"source": "file.txt", "page": None, "index": 1}
    result = _sanitize_metadata(meta)
    assert result["page"] == "None"   # converted to string, not dropped
    assert result["source"] == "file.txt"
    assert result["index"] == 1


def test_sanitize_metadata_keeps_valid_types():
    meta = {"a": "string", "b": 42, "c": 3.14, "d": True}
    result = _sanitize_metadata(meta)
    assert result == meta