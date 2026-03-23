import chromadb
from chromadb.config import Settings
from pathlib import Path
import numpy as np
import uuid


# ─────────────────────────────────────────────
# VECTOR STORE CLASS
# ─────────────────────────────────────────────

class VectorStore:
    """
    Wraps ChromaDB — a local, persistent vector database.

    Why ChromaDB?
        - Runs fully on disk, no server needed
        - Free and open source
        - Supports metadata filtering (filter by source, page, type)
        - Simple Python API

    Folder structure after first use:
        p1-ingestion/
        └── chroma_db/          ← auto-created, in .gitignore
            └── <collection>/
                ├── data_level0.bin
                └── header.bin
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "documents",
    ):
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # PersistentClient saves data to disk automatically
        # Every add/upsert is immediately persisted — no manual .save() needed
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # A collection is like a table in a relational DB
        # get_or_create means we can re-run the pipeline safely
        # without wiping data every time
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity for search
        )

        self.collection_name = collection_name
        print(
            f"[VECTOR STORE] Collection '{collection_name}' ready. "
            f"Documents stored: {self.collection.count()}"
        )

    # ─────────────────────────────────────────
    # ADD
    # ─────────────────────────────────────────

    def add(self, embedded_chunks: list[dict]) -> int:
        """
        Stores embedded chunks into ChromaDB.

        Each item in embedded_chunks must have:
            - "content":   str
            - "embedding": np.ndarray
            - "metadata":  dict

        Returns the number of chunks actually added.

        Why upsert instead of add?
            If you run the ingestion pipeline twice on the same file,
            upsert updates existing records instead of throwing a
            duplicate ID error. Safer for re-runs.
        """
        if not embedded_chunks:
            print("[VECTOR STORE] Nothing to add.")
            return 0

        ids        = []
        documents  = []
        embeddings = []
        metadatas  = []

        for item in embedded_chunks:
            # Generate a stable ID from source + chunk_index
            # This means re-ingesting the same file gives the same IDs
            # so upsert correctly updates rather than duplicates
            source      = item["metadata"].get("source", "unknown")
            chunk_index = item["metadata"].get("chunk_index", 0)
            stable_id   = _make_id(source, chunk_index)

            ids.append(stable_id)
            documents.append(item["content"])
            embeddings.append(item["embedding"].tolist())  # ChromaDB needs plain list
            metadatas.append(_sanitize_metadata(item["metadata"]))

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[VECTOR STORE] Upserted {len(ids)} chunks. "
              f"Total in collection: {self.collection.count()}")
        return len(ids)

    # ─────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Finds the top_k most similar chunks to a query vector.

        Args:
            query_embedding: vector from Embedder.embed_text(query)
            top_k:           how many results to return
            where:           optional metadata filter, e.g. {"type": "pdf"}

        Returns a list of dicts:
            [
                {
                    "content":  "chunk text...",
                    "metadata": { source, type, chunk_index, ... },
                    "score":    0.87   # cosine similarity, higher = better
                },
                ...
            ]
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results":        min(top_k, self.collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        if self.collection.count() == 0:
            print("[VECTOR STORE] Collection is empty — nothing to query.")
            return []

        results = self.collection.query(**query_params)

        # ChromaDB returns distances (lower = more similar for cosine space)
        # Convert to similarity scores (higher = more similar) for clarity
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "content":  doc,
                "metadata": meta,
                "score":    round(1 - dist, 4),  # cosine distance → similarity
            })

        return output

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def count(self) -> int:
        """Returns total number of chunks stored in the collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """
        Wipes the entire collection.
        Useful for tests and fresh re-ingestion runs.
        """
        self.client.delete_collection(self.collection_name)
        print(f"[VECTOR STORE] Collection '{self.collection_name}' deleted.")

    def get_all_sources(self) -> list[str]:
        """
        Returns a deduplicated list of all source file paths/URLs stored.
        Useful for showing the user what's been ingested.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in results["metadatas"]}
        return sorted(sources)


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _make_id(source: str, chunk_index: int) -> str:
    """
    Creates a stable, deterministic ID for a chunk.
    Same source + same index always produces the same ID.
    This is what makes upsert safe for re-runs.
    """
    raw = f"{source}::chunk::{chunk_index}"
    # Use uuid5 — deterministic UUID derived from a string
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB only accepts metadata values of type str, int, float, or bool.
    This strips out anything else (like None) that would cause an error.
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)  # convert anything else to string
    return sanitized