import pytest
import numpy as np
from src.loaders import Document
from src.chunker import Chunk, chunk_document
from src.embedder import Embedder, SUPPORTED_MODELS, DEFAULT_MODEL


# ── fixture — load the model once for all tests in this file ──
# This is a pytest fixture — it runs once per test session,
# not once per test. Without this, each test would reload the model.

@pytest.fixture(scope="module")
def embedder():
    return Embedder(model_key=DEFAULT_MODEL)


def _make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        content=text,
        metadata={"source": "test.txt", "type": "txt", "chunk_index": index}
    )


# ── model loading tests ───────────────────────────────────

def test_embedder_loads_successfully(embedder):
    assert embedder.model is not None


def test_embedder_correct_model_name(embedder):
    assert embedder.model_name == SUPPORTED_MODELS[DEFAULT_MODEL]


def test_embedder_embedding_dim_positive(embedder):
    assert embedder.embedding_dim > 0


def test_invalid_model_key_raises():
    with pytest.raises(ValueError, match="Unknown model key"):
        Embedder(model_key="gpt-99-fake")


# ── embed_text tests ──────────────────────────────────────

def test_embed_text_returns_numpy_array(embedder):
    vec = embedder.embed_text("What is retrieval augmented generation?")
    assert isinstance(vec, np.ndarray)


def test_embed_text_correct_shape(embedder):
    vec = embedder.embed_text("Hello world")
    assert vec.shape == (embedder.embedding_dim,)


def test_embed_text_is_normalized(embedder):
    vec = embedder.embed_text("Normalize this sentence.")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5   # unit vector — norm should be ~1.0


def test_embed_text_empty_raises(embedder):
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_text("")


def test_embed_text_whitespace_raises(embedder):
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_text("   ")


# ── embed_chunks tests ────────────────────────────────────

def test_embed_chunks_returns_list(embedder):
    chunks = [_make_chunk("RAG is a retrieval technique.", 0)]
    result = embedder.embed_chunks(chunks, show_progress=False)
    assert isinstance(result, list)


def test_embed_chunks_correct_count(embedder):
    chunks = [_make_chunk(f"Sentence number {i}.", i) for i in range(5)]
    result = embedder.embed_chunks(chunks, show_progress=False)
    assert len(result) == 5


def test_embed_chunks_has_required_keys(embedder):
    chunks = [_make_chunk("Test chunk content.", 0)]
    result = embedder.embed_chunks(chunks, show_progress=False)
    assert "content" in result[0]
    assert "embedding" in result[0]
    assert "metadata" in result[0]


def test_embed_chunks_embedding_shape(embedder):
    chunks = [_make_chunk("Another test sentence.", 0)]
    result = embedder.embed_chunks(chunks, show_progress=False)
    assert result[0]["embedding"].shape == (embedder.embedding_dim,)


def test_embed_chunks_metadata_preserved(embedder):
    chunks = [_make_chunk("Metadata test.", 0)]
    result = embedder.embed_chunks(chunks, show_progress=False)
    assert result[0]["metadata"]["source"] == "test.txt"
    assert result[0]["metadata"]["chunk_index"] == 0


def test_embed_chunks_empty_input(embedder):
    result = embedder.embed_chunks([], show_progress=False)
    assert result == []


# ── cosine similarity tests ───────────────────────────────

def test_cosine_similarity_identical_vectors(embedder):
    vec = embedder.embed_text("The cat sat on the mat.")
    score = embedder.cosine_similarity(vec, vec)
    assert abs(score - 1.0) < 1e-5   # same vector = similarity of 1.0


def test_cosine_similarity_similar_sentences(embedder):
    """
    Semantically similar sentences should score higher than unrelated ones.
    This is the core promise of a good embedding model.
    """
    vec_a = embedder.embed_text("How does retrieval augmented generation work?")
    vec_b = embedder.embed_text("Explain the RAG technique in NLP.")
    vec_c = embedder.embed_text("What is the boiling point of water?")

    similar_score   = embedder.cosine_similarity(vec_a, vec_b)
    dissimilar_score = embedder.cosine_similarity(vec_a, vec_c)

    assert similar_score > dissimilar_score


def test_cosine_similarity_range(embedder):
    vec_a = embedder.embed_text("Machine learning is fascinating.")
    vec_b = embedder.embed_text("I enjoy cooking pasta.")
    score = embedder.cosine_similarity(vec_a, vec_b)
    assert -1.0 <= score <= 1.0


def test_cosine_similarity_zero_vector(embedder):
    vec_a = embedder.embed_text("Some text.")
    zero  = np.zeros(embedder.embedding_dim)
    score = embedder.cosine_similarity(vec_a, zero)
    assert score == 0.0


# ── integration test — chunk → embed pipeline ─────────────

def test_full_chunk_to_embed_pipeline(embedder):
    """
    Simulates the real pipeline:
    Document → chunks → embedded dicts ready for vector store.
    """
    doc = Document(
        content=(
            "Vector databases store high-dimensional embeddings. "
            "They allow fast similarity search over millions of vectors. "
            "ChromaDB is a popular open-source vector store. "
            "It persists data to disk and runs entirely locally."
        ),
        metadata={"source": "test_doc.txt", "type": "txt", "filename": "test_doc.txt"}
    )
    chunks  = chunk_document(doc, chunk_size=120, chunk_overlap=20)
    results = embedder.embed_chunks(chunks, show_progress=False)

    assert len(results) > 0
    for r in results:
        assert r["embedding"].shape[0] == embedder.embedding_dim
        assert r["metadata"]["source"] == "test_doc.txt"