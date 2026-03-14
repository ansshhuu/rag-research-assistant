import pytest
from pathlib import Path
from src.loaders import load_pdf, load_text_file, load_document, Document

# ── helpers ──────────────────────────────────────────────
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_docs"


def _make_txt(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return str(f)


# ── text loader tests ─────────────────────────────────────

def test_load_text_returns_documents(tmp_path):
    path = _make_txt(tmp_path, "hello.txt", "Hello world. This is a test.")
    docs = load_text_file(path)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)


def test_load_text_content_correct(tmp_path):
    path = _make_txt(tmp_path, "content.txt", "RAG is cool.")
    docs = load_text_file(path)
    assert "RAG is cool." in docs[0].content


def test_load_text_metadata_has_source(tmp_path):
    path = _make_txt(tmp_path, "meta.txt", "some content")
    docs = load_text_file(path)
    assert "source" in docs[0].metadata
    assert "meta.txt" in docs[0].metadata["source"]


def test_load_text_markdown(tmp_path):
    path = _make_txt(tmp_path, "notes.md", "# Title\n\nSome notes here.")
    docs = load_text_file(path)
    assert docs[0].metadata["type"] == "md"


def test_load_text_empty_file_raises(tmp_path):
    path = _make_txt(tmp_path, "empty.txt", "")
    with pytest.raises(ValueError, match="empty"):
        load_text_file(path)


def test_load_text_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_text_file("does_not_exist.txt")


def test_load_text_wrong_extension_raises(tmp_path):
    f = tmp_path / "file.csv"
    f.write_text("a,b,c")
    with pytest.raises(ValueError, match=".csv"):
        load_text_file(str(f))


# ── unified loader routing tests ──────────────────────────

def test_load_document_routes_txt(tmp_path):
    path = _make_txt(tmp_path, "route.txt", "routing test")
    docs = load_document(str(path))
    assert docs[0].metadata["type"] == "txt"


def test_load_document_routes_md(tmp_path):
    path = _make_txt(tmp_path, "route.md", "# heading")
    docs = load_document(str(path))
    assert docs[0].metadata["type"] == "md"


def test_load_document_unsupported_raises(tmp_path):
    f = tmp_path / "file.json"
    f.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(str(f))