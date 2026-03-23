import numpy as np
from sentence_transformers import SentenceTransformer
from src.chunker import Chunk
from tqdm import tqdm



SUPPORTED_MODELS = {
    "minilm":   "all-MiniLM-L6-v2",       # 80MB  — fast, good quality, default
    "mpnet":    "all-mpnet-base-v2",       # 420MB — slower, better quality
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # multi-language
}

DEFAULT_MODEL = "minilm"



class Embedder:
    def __init__(self, model_key: str = DEFAULT_MODEL):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Choose from: {list(SUPPORTED_MODELS.keys())}"
            )

        model_name = SUPPORTED_MODELS[model_key]
        print(f"[EMBEDDER] Loading model '{model_name}' ...")

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
        return vector 

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> list[dict]:
    
        if not chunks:
            print("[EMBEDDER] No chunks to embed.")
            return []

        texts = [chunk.content for chunk in chunks]

        print(f"[EMBEDDER] Embedding {len(texts)} chunks in batches of {batch_size}...")

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   
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
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
