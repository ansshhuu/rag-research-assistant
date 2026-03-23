import numpy as np
from sentence_transformers import SentenceTransformer
from src.chunker import Chunk
from tqdm import tqdm


# ─────────────────────────────────────────────
# MODEL REGISTRY — swap model here, nothing else changes
# ─────────────────────────────────────────────

SUPPORTED_MODELS = {
    "minilm":   "all-MiniLM-L6-v2",       # 80MB  — fast, good quality, default
    "mpnet":    "all-mpnet-base-v2",       # 420MB — slower, better quality
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # multi-language
}

DEFAULT_MODEL = "minilm"


# ─────────────────────────────────────────────
# EMBEDDER CLASS
# ─────────────────────────────────────────────

class Embedder:
    """
    Wraps a sentence-transformers model.
    Loads the model once, reuses it for all encode calls.

    Why a class and not just a function?
        Loading the model takes ~1-2 seconds. If it were a plain function,
        every call would reload the model. A class loads it once in __init__
        and keeps it in memory for the lifetime of the pipeline run.
    """

    def __init__(self, model_key: str = DEFAULT_MODEL):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Choose from: {list(SUPPORTED_MODELS.keys())}"
            )

        model_name = SUPPORTED_MODELS[model_key]
        print(f"[EMBEDDER] Loading model '{model_name}' ...")

        # Downloads model on first use, caches it locally after that
        # Cache location: C:\Users\<you>\.cache\torch\sentence_transformers
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.model_key = model_key
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"[EMBEDDER] Ready. Embedding dim = {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single string → returns a 1D numpy array.
        Used at query time in P2 — embed the user's question.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        vector = self.model.encode(text, normalize_embeddings=True)
        return vector  # shape: (embedding_dim,)

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Embeds a list of Chunks in batches.
        Returns a list of dicts — each dict has the vector + full metadata.
        This is exactly what vector_store.py will receive.

        Why batching?
            Encoding one chunk at a time is slow. The model processes a
            batch of 64 in roughly the same time as a single chunk.
            For 1000 chunks, batching gives ~15x speedup.

        Returns:
            [
                {
                    "content":   "chunk text...",
                    "embedding": np.array([0.12, -0.05, ...]),  # shape (384,)
                    "metadata":  { source, type, chunk_index, ... }
                },
                ...
            ]
        """
        if not chunks:
            print("[EMBEDDER] No chunks to embed.")
            return []

        texts = [chunk.content for chunk in chunks]

        print(f"[EMBEDDER] Embedding {len(texts)} chunks in batches of {batch_size}...")

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # makes cosine similarity = dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embedded = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append({
                "content":   chunk.content,
                "embedding": vector,
                "metadata":  chunk.metadata,
            })

        print(f"[EMBEDDER] Done. {len(embedded)} embeddings, dim={vectors.shape[1]}")
        return embedded

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Computes similarity between two vectors.
        Returns a float between -1 and 1. Higher = more similar.

        Note: since we use normalize_embeddings=True above,
        vectors already have unit length, so this is just a dot product.
        Keeping the explicit formula here for clarity and testability.
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))