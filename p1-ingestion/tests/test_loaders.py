import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from src.loaders import (
    load_pdf,
    load_text_file,
    load_document,
    load_webpage,
    Document,
)


# ── helpers ──────────────────────────────────────────────

def _make_txt(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return str(f)


# ── TEXT LOADER TESTS ─────────────────────────────────────

def test_load_text_returns_documents(tmp_path):
    path = _make_txt(tmp_path, "hello.txt", "Hello world.")
    docs = load_text_file(path)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)


def test_load_text_content_correct(tmp_path):
    path = _make_txt(tmp_path, "content.txt", "RAG is cool.")
    docs = load_text_file(path)
    assert "RAG is cool." in docs[0].content


def test_load_text_metadata_has_source(tmp_path):
    path = _make_txt(tmp_path, "meta.txt", "data")
    docs = load_text_file(path)
    assert "source" in docs[0].metadata
    assert "meta.txt" in docs[0].metadata["source"]


def test_load_text_markdown(tmp_path):
    path = _make_txt(tmp_path, "notes.md", "# Title")
    docs = load_text_file(path)
    assert docs[0].metadata["type"] == "md"


def test_load_text_empty_file_raises(tmp_path):
    path = _make_txt(tmp_path, "empty.txt", "")
    with pytest.raises(ValueError):
        load_text_file(path)


def test_load_text_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_text_file("missing.txt")


def test_load_text_wrong_extension_raises(tmp_path):
    f = tmp_path / "file.csv"
    f.write_text("a,b,c")
    with pytest.raises(ValueError):
        load_text_file(str(f))


# ── PDF LOADER TESTS ─────────────────────────────────────

def test_load_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("fake.pdf")


def test_load_pdf_wrong_extension(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("not pdf")
    with pytest.raises(ValueError):
        load_pdf(str(f))


# ── WEB LOADER TESTS (MOCKED) ────────────────────────────

def test_load_webpage_success():
    html = "<html><title>Test</title><body>Hello world</body></html>"

    with patch("requests.get") as mock_get:
        mock_res = Mock()
        mock_res.text = html
        mock_res.raise_for_status = Mock()
        mock_get.return_value = mock_res

        docs = load_webpage("http://test.com")

        assert len(docs) == 1
        assert "Hello world" in docs[0].content
        assert docs[0].metadata["type"] == "webpage"


def test_load_webpage_failure():
    with patch("requests.get", side_effect=Exception("fail")):
        with pytest.raises(Exception):
            load_webpage("http://fail.com")


# ── UNIFIED LOADER TESTS ─────────────────────────────────

def test_load_document_routes_txt(tmp_path):
    path = _make_txt(tmp_path, "file.txt", "text")
    docs = load_document(path)
    assert docs[0].metadata["type"] == "txt"


def test_load_document_routes_md(tmp_path):
    path = _make_txt(tmp_path, "file.md", "# md")
    docs = load_document(path)
    assert docs[0].metadata["type"] == "md"


def test_load_document_routes_url():
    with patch("src.loaders.load_webpage") as mock_loader:
        mock_loader.return_value = []
        load_document("http://example.com")
        mock_loader.assert_called_once()


def test_load_document_empty_source():
    with pytest.raises(ValueError):
        load_document("")


def test_load_document_unsupported_raises(tmp_path):
    f = tmp_path / "file.json"
    f.write_text("{}")
    with pytest.raises(ValueError):
        load_document(str(f))