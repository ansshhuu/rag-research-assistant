import chromadb
from chromadb.config import Settings
from pathlib import Path
import numpy as np
import uuid

class VectorStore:
    
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "documents",
    ):
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}, 
        )

        self.collection_name = collection_name
        print(
            f"[VECTOR STORE] Collection '{collection_name}' ready. "
            f"Documents stored: {self.collection.count()}"
        )
    def add(self, embedded_chunks: list[dict]) -> int:
        if not embedded_chunks:
            print("[VECTOR STORE] Nothing to add.")
            return 0

        ids        = []
        documents  = []
        embeddings = []
        metadatas  = []

        for item in embedded_chunks:
            source      = item["metadata"].get("source", "unknown")
            chunk_index = item["metadata"].get("chunk_index", 0)
            stable_id   = _make_id(source, chunk_index)

            ids.append(stable_id)
            documents.append(item["content"])
            embeddings.append(item["embedding"].tolist()) 
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
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
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
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "content":  doc,
                "metadata": meta,
                "score":    round(1 - dist, 4), 
            })

        return output


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



def _make_id(source: str, chunk_index: int) -> str:
    
    raw = f"{source}::chunk::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _sanitize_metadata(metadata: dict) -> dict:

    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value) 
    return sanitized
