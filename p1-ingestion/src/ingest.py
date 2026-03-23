import argparse
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from src.loaders import load_document
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vector_store import VectorStore

load_dotenv()


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_ingestion(
    sources: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    model_key: str = "minilm",
    persist_directory: str = "chroma_db",
    collection_name: str = "documents",
) -> dict:
    """
    The full ingestion pipeline — end to end:

        sources → load → chunk → embed → store

    Returns a summary dict so callers (and tests) can inspect what happened.
    """
    start_time = time.time()
    summary = {
        "sources_requested": len(sources),
        "sources_succeeded": 0,
        "sources_failed":    0,
        "total_documents":   0,
        "total_chunks":      0,
        "total_embedded":    0,
        "failed_sources":    [],
        "duration_seconds":  0.0,
    }

    # ── Step 1: Load ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 1 — LOADING DOCUMENTS")
    print("=" * 55)

    all_documents = []

    for source in sources:
        try:
            docs = load_document(source)
            all_documents.extend(docs)
            summary["sources_succeeded"] += 1
        except Exception as e:
            print(f"[ERROR] Failed to load '{source}': {e}")
            summary["sources_failed"] += 1
            summary["failed_sources"].append(source)

    summary["total_documents"] = len(all_documents)

    if not all_documents:
        print("\n[PIPELINE] No documents loaded. Exiting.")
        return summary

    # ── Step 2: Chunk ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2 — CHUNKING")
    print("=" * 55)

    chunks = chunk_documents(
        all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    summary["total_chunks"] = len(chunks)

    if not chunks:
        print("\n[PIPELINE] No chunks produced. Exiting.")
        return summary

    # ── Step 3: Embed ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3 — EMBEDDING")
    print("=" * 55)

    embedder = Embedder(model_key=model_key)
    embedded_chunks = embedder.embed_chunks(chunks)
    summary["total_embedded"] = len(embedded_chunks)

    # ── Step 4: Store ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4 — STORING IN VECTOR DB")
    print("=" * 55)

    store = VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    store.add(embedded_chunks)

    # ── Summary ───────────────────────────────────────────
    summary["duration_seconds"] = round(time.time() - start_time, 2)

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Sources requested : {summary['sources_requested']}")
    print(f"  Sources succeeded : {summary['sources_succeeded']}")
    print(f"  Sources failed    : {summary['sources_failed']}")
    print(f"  Documents loaded  : {summary['total_documents']}")
    print(f"  Chunks created    : {summary['total_chunks']}")
    print(f"  Chunks embedded   : {summary['total_embedded']}")
    print(f"  Duration          : {summary['duration_seconds']}s")
    print("=" * 55 + "\n")

    return summary


# ─────────────────────────────────────────────
# QUERY HELPER — lets you test retrieval from CLI
# ─────────────────────────────────────────────

def run_query(
    query_text: str,
    top_k: int = 5,
    model_key: str = "minilm",
    persist_directory: str = "chroma_db",
    collection_name: str = "documents",
    where: dict | None = None,
) -> list[dict]:
    """
    Embeds a query and retrieves the top_k most relevant chunks.
    This is a preview of what P2's RAG engine will do properly.
    """
    embedder = Embedder(model_key=model_key)
    store    = VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    query_vector = embedder.embed_text(query_text)
    results      = store.query(query_vector, top_k=top_k, where=where)

    print(f"\nQuery: '{query_text}'")
    print(f"Top {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r['score']}  |  Source: {r['metadata'].get('source', 'unknown')}")
        print(f"       {r['content'][:200]}...")
        print()

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG Research Assistant — Document Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a PDF
  python -m src.ingest ingest --sources data/sample_docs/paper.pdf

  # Ingest multiple sources
  python -m src.ingest ingest --sources file.pdf notes.txt https://example.com

  # Ingest with custom chunk settings
  python -m src.ingest ingest --sources paper.pdf --chunk-size 300 --overlap 30

  # Query after ingestion
  python -m src.ingest query --text "What is retrieval augmented generation?"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest subcommand ─────────────────────
    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion pipeline")
    ingest_parser.add_argument(
        "--sources", nargs="+", required=True,
        help="One or more file paths or URLs to ingest"
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Max characters per chunk (default: 500)"
    )
    ingest_parser.add_argument(
        "--overlap", type=int, default=50,
        help="Overlap characters between chunks (default: 50)"
    )
    ingest_parser.add_argument(
        "--model", type=str, default="minilm",
        choices=["minilm", "mpnet", "multilingual"],
        help="Embedding model to use (default: minilm)"
    )
    ingest_parser.add_argument(
        "--collection", type=str, default="documents",
        help="ChromaDB collection name (default: documents)"
    )

    # ── query subcommand ──────────────────────
    query_parser = subparsers.add_parser("query", help="Query the vector store")
    query_parser.add_argument(
        "--text", type=str, required=True,
        help="Query text to search for"
    )
    query_parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)"
    )
    query_parser.add_argument(
        "--model", type=str, default="minilm",
        choices=["minilm", "mpnet", "multilingual"],
        help="Embedding model to use (default: minilm)"
    )
    query_parser.add_argument(
        "--collection", type=str, default="documents",
        help="ChromaDB collection name (default: documents)"
    )
    query_parser.add_argument(
        "--filter-type", type=str, default=None,
        choices=["pdf", "txt", "md", "webpage"],
        help="Filter results by document type"
    )

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "ingest":
        summary = run_ingestion(
            sources=args.sources,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            model_key=args.model,
            collection_name=args.collection,
        )
        if summary["sources_failed"] > 0:
            sys.exit(1)   # non-zero exit = CI pipeline knows something failed

    elif args.command == "query":
        where = {"type": args.filter_type} if args.filter_type else None
        run_query(
            query_text=args.text,
            top_k=args.top_k,
            model_key=args.model,
            collection_name=args.collection,
            where=where,
        )


if __name__ == "__main__":
    main()