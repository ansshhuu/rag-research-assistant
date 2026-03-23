import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.ingest import run_ingestion, run_query, build_parser


# ── helpers ───────────────────────────────────────────────

def _make_txt(tmp_path: Path, name: str, content: str) -> str:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return str(f)


SAMPLE_TEXT = (
    "Retrieval Augmented Generation combines search with generation. "
    "It retrieves relevant documents from a vector store first. "
    "Then it uses those documents as context for the language model. "
    "This approach reduces hallucinations significantly. "
    "Vector databases store high-dimensional embeddings efficiently. "
    "ChromaDB is a popular open-source option that runs locally. "
)


# ── pipeline integration tests ────────────────────────────

def test_ingest_single_txt_file(tmp_path):
    source = _make_txt(tmp_path, "doc.txt", SAMPLE_TEXT)
    summary = run_ingestion(
        sources=[source],
        chunk_size=200,
        chunk_overlap=20,
        persist_directory=str(tmp_path / "chroma"),
    )
    assert summary["sources_succeeded"] == 1
    assert summary["sources_failed"]    == 0
    assert summary["total_documents"]   >= 1
    assert summary["total_chunks"]      >= 1
    assert summary["total_embedded"]    >= 1


def test_ingest_multiple_files(tmp_path):
    source_a = _make_txt(tmp_path, "a.txt", SAMPLE_TEXT)
    source_b = _make_txt(tmp_path, "b.txt", SAMPLE_TEXT + " Extra content here.")
    summary  = run_ingestion(
        sources=[source_a, source_b],
        chunk_size=200,
        chunk_overlap=20,
        persist_directory=str(tmp_path / "chroma"),
    )
    assert summary["sources_succeeded"] == 2
    assert summary["total_chunks"]      >= 2


def test_ingest_bad_source_is_recorded(tmp_path):
    """
    A missing file should fail gracefully — not crash the whole pipeline.
    The summary should record the failure and the good sources still succeed.
    """
    good   = _make_txt(tmp_path, "good.txt", SAMPLE_TEXT)
    bad    = str(tmp_path / "does_not_exist.txt")
    summary = run_ingestion(
        sources=[good, bad],
        chunk_size=200,
        chunk_overlap=20,
        persist_directory=str(tmp_path / "chroma"),
    )
    assert summary["sources_succeeded"] == 1
    assert summary["sources_failed"]    == 1
    assert bad in summary["failed_sources"]


def test_ingest_all_bad_sources_returns_empty_summary(tmp_path):
    summary = run_ingestion(
        sources=["fake1.txt", "fake2.txt"],
        chunk_size=200,
        chunk_overlap=20,
        persist_directory=str(tmp_path / "chroma"),
    )
    assert summary["total_documents"] == 0
    assert summary["total_chunks"]    == 0
    assert summary["total_embedded"]  == 0


def test_ingest_summary_has_duration(tmp_path):
    source  = _make_txt(tmp_path, "timing.txt", SAMPLE_TEXT)
    summary = run_ingestion(
        sources=[source],
        chunk_size=200,
        chunk_overlap=20,
        persist_directory=str(tmp_path / "chroma"),
    )
    assert summary["duration_seconds"] > 0


def test_ingest_is_idempotent(tmp_path):
    """
    Running the pipeline twice on the same file should NOT
    double the chunk count — upsert must handle duplicates.
    """
    source   = _make_txt(tmp_path, "idem.txt", SAMPLE_TEXT)
    chroma   = str(tmp_path / "chroma")

    summary1 = run_ingestion(
        sources=[source], chunk_size=200, chunk_overlap=20,
        persist_directory=chroma,
    )
    summary2 = run_ingestion(
        sources=[source], chunk_size=200, chunk_overlap=20,
        persist_directory=chroma,
    )
    assert summary1["total_chunks"] == summary2["total_chunks"]


# ── query integration tests ───────────────────────────────

def test_query_returns_results_after_ingestion(tmp_path):
    source = _make_txt(tmp_path, "query_doc.txt", SAMPLE_TEXT)
    chroma = str(tmp_path / "chroma")

    run_ingestion(
        sources=[source], chunk_size=200, chunk_overlap=20,
        persist_directory=chroma,
    )

    results = run_query(
        query_text="What is retrieval augmented generation?",
        top_k=3,
        persist_directory=chroma,
    )
    assert len(results) > 0
    assert all("content"  in r for r in results)
    assert all("score"    in r for r in results)
    assert all("metadata" in r for r in results)


def test_query_scores_are_valid(tmp_path):
    source = _make_txt(tmp_path, "scores.txt", SAMPLE_TEXT)
    chroma = str(tmp_path / "chroma")
    run_ingestion(sources=[source], chunk_size=200, chunk_overlap=20,
                  persist_directory=chroma)

    results = run_query("vector database", top_k=3, persist_directory=chroma)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0


# ── CLI parser tests ──────────────────────────────────────

def test_parser_ingest_command():
    parser = build_parser()
    args   = parser.parse_args(["ingest", "--sources", "file.txt"])
    assert args.command    == "ingest"
    assert args.sources    == ["file.txt"]
    assert args.chunk_size == 500     # default
    assert args.overlap    == 50      # default
    assert args.model      == "minilm"


def test_parser_ingest_custom_flags():
    parser = build_parser()
    args   = parser.parse_args([
        "ingest", "--sources", "a.pdf", "b.txt",
        "--chunk-size", "300",
        "--overlap", "30",
        "--model", "mpnet",
    ])
    assert args.sources    == ["a.pdf", "b.txt"]
    assert args.chunk_size == 300
    assert args.overlap    == 30
    assert args.model      == "mpnet"


def test_parser_query_command():
    parser = build_parser()
    args   = parser.parse_args(["query", "--text", "What is RAG?"])
    assert args.command == "query"
    assert args.text    == "What is RAG?"
    assert args.top_k   == 5


def test_parser_query_with_filter():
    parser = build_parser()
    args   = parser.parse_args([
        "query", "--text", "embeddings", "--filter-type", "pdf"
    ])
    assert args.filter_type == "pdf"


def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])