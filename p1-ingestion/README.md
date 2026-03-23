# P1 — Document Ingestion Pipeline

> Load any document (PDF, webpage, text) → chunk it → embed it → store it in a vector database.  
> This is the foundation of the entire RAG Research Assistant system.

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Ingestion Pipeline                        │
│                                                             │
│  [Source]──►[Loader]──►[Chunker]──►[Embedder]──►[VectorDB] │
│                                                             │
│  .pdf                  500 chars    MiniLM-L6    ChromaDB   │
│  .txt     load_document chunk_docs  local model  persisted  │
│  .md                   + overlap    384-dim vec  on disk    │
│  URL                                                        │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Tool | Purpose | Cost |
|---|---|---|
| PyMuPDF | PDF text extraction | Free |
| BeautifulSoup4 | Web page scraping | Free |
| LangChain | Text splitting utilities | Free |
| sentence-transformers | Local embedding model | Free |
| ChromaDB | Local vector database | Free |
| pytest | Testing | Free |

**Zero paid APIs. Runs fully offline after first model download.**

## Quickstart
```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest a document
python -m src.ingest ingest --sources data/sample_docs/paper.pdf

# 4. Ingest multiple sources at once
python -m src.ingest ingest --sources paper.pdf notes.txt https://en.wikipedia.org/wiki/RAG

# 5. Query what you ingested
python -m src.ingest query --text "What is retrieval augmented generation?" --top-k 3

# 6. Filter by document type
python -m src.ingest query --text "embedding models" --filter-type pdf
```

## CLI Reference
```
ingest subcommand:
  --sources       One or more file paths or URLs  (required)
  --chunk-size    Characters per chunk            (default: 500)
  --overlap       Overlap between chunks          (default: 50)
  --model         minilm | mpnet | multilingual   (default: minilm)
  --collection    ChromaDB collection name        (default: documents)

query subcommand:
  --text          Query string                    (required)
  --top-k         Number of results               (default: 5)
  --model         minilm | mpnet | multilingual   (default: minilm)
  --filter-type   pdf | txt | md | webpage        (optional)
  --collection    ChromaDB collection name        (default: documents)
```

## Project Structure
```
p1-ingestion/
├── src/
│   ├── loaders.py        # PDF, web, text file loaders → Document
│   ├── chunker.py        # Sliding window chunker → Chunk
│   ├── embedder.py       # sentence-transformers wrapper → vectors
│   ├── vector_store.py   # ChromaDB wrapper → store + query
│   └── ingest.py         # Pipeline orchestrator + CLI
├── tests/
│   ├── test_loaders.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   └── test_ingest.py
├── data/
│   └── sample_docs/      # drop your files here (gitignored)
├── chroma_db/            # auto-created on first run (gitignored)
├── requirements.txt
├── for-learn.md          # concepts learned building this
└── README.md
```

## Running Tests
```bash
# Run all tests with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a single test file
pytest tests/test_chunker.py -v
```

## Key Design Decisions

**Why character-based chunking?**  
Token-based chunking requires a tokenizer tied to a specific model. Character-based chunking is model-agnostic and portable. For P1, this is the right trade-off. Token-based chunking will be revisited in P3.

**Why `all-MiniLM-L6-v2`?**  
80MB, runs on CPU, 384-dim embeddings, strong performance on semantic similarity benchmarks. No API key, no cost, no rate limits. Upgrade path to `all-mpnet-base-v2` (420MB, better quality) is one flag change: `--model mpnet`.

**Why ChromaDB over FAISS?**  
ChromaDB persists to disk automatically, supports metadata filtering, and has a cleaner Python API. FAISS is faster at scale but requires manual serialization and has no metadata support. For a research assistant with <1M chunks, ChromaDB is the correct choice.

**Why upsert over insert?**  
Idempotency. Running the pipeline twice on the same file should update records, not duplicate them. Stable IDs derived from `source + chunk_index` via `uuid5` guarantee this.