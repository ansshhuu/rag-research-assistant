# What I learned building P1

## 1. Chunking

**The core trade-off:**  
- Chunk too small → each chunk loses surrounding context, retrieval returns fragments
- Chunk too large → each chunk contains too many topics, similarity score gets diluted

**What I tried:**  
- `chunk_size=200, overlap=0` — chunks felt disconnected at boundaries
- `chunk_size=500, overlap=50` — noticeably better retrieval on test queries
- `chunk_size=500, overlap=0` vs `overlap=50` — with overlap, queries near chunk boundaries returned complete answers

**Sentence boundary detection:**  
Hard-cutting at exactly N characters splits sentences mid-word. Walking backwards up to 80 chars to find a `. `, `? ` or `! ` produces cleaner chunks. Better sentence boundaries = better embeddings = better retrieval.

**What I'd do differently at scale:**  
Use semantic chunking — split on topic shifts detected by the embedding model itself, not on character counts. LangChain has a `SemanticChunker` for this.

---

## 2. Embeddings

**What an embedding actually is:**  
A vector is a point in high-dimensional space. Two sentences that mean similar things end up close together in that space. `all-MiniLM-L6-v2` maps any text to a 384-dimensional point.

**Why normalize embeddings:**  
Normalizing to unit length makes cosine similarity equal to dot product. Dot product is cheaper to compute and what ChromaDB uses internally. Without normalization, similarity scores are inconsistent.

**Dense vs sparse vectors:**  
- Dense (what we use): every dimension has a value, captures semantic meaning
- Sparse (BM25, TF-IDF): most dimensions are zero, captures keyword matches
- Hybrid search combines both — a P3/P4 improvement worth knowing about

**Model comparison I did:**

| Model | Size | Dim | Speed | Quality |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 80MB | 384 | Fast | Good |
| all-mpnet-base-v2 | 420MB | 768 | Slow | Better |

For a local pipeline, MiniLM is the right default.

---

## 3. Vector databases

**What ChromaDB actually stores:**  
For each chunk: the text content, the embedding vector (384 floats), and a metadata dict. The HNSW index makes nearest-neighbor search fast even with 100k+ vectors.

**Why metadata matters:**  
Without metadata, you get a matching chunk but no idea where it came from. With `source`, `page`, `type` on every chunk, the query result tells you exactly which document and page to cite.

**Idempotency via uuid5:**  
`uuid5` generates a deterministic UUID from a string. Same `source + chunk_index` → same ID → upsert updates instead of duplicating. This is a production data pipeline pattern.

**HNSW distance → similarity:**  
ChromaDB returns cosine *distance* (lower = more similar). Subtracting from 1 gives cosine *similarity* (higher = more similar). Easy to confuse — always check the docs.

---

