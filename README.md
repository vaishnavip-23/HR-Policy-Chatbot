# Multi-Doc HR Policy RAG System

Advanced RAG system for querying multiple HR policy documents using hybrid search, query expansion, and local reranking.

## 🎯 Features

- **Multi-Document Search**: Query across 3 HR policy documents (IIMA, Chemexcil, TCCAP)
- **Hybrid Retrieval**: Combines BM25 (keyword) + Vector (semantic) search
- **Query Expansion**: Generates 3 query variations for better recall
- **Local Reranker**: FREE cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- **LLM Answer Generation**: GPT-5-mini with citations and confidence scoring
- **Streamlit UI**: Beautiful chat interface with configuration options
- **Full Text Context**: Uses complete chunk text (not summaries) for accurate answers

## 📊 System Architecture

```
Query → Query Translation (1→4 variants)
  ↓
Hybrid Search (per variant):
  ├─ BM25 keyword search → 5 results
  └─ Vector semantic search → 5 results
  = 40 total results (20 BM25 + 20 Vector)
  ↓
RRF Fusion → Top 30 unique chunks
  ↓
Local Cross-Encoder Reranking → Top 10 chunks
  ↓
LLM Answer Generation → Answer + Citations
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Setup Environment

Create `.env` file:

```bash
# Required
OPENAI_API_KEY=your-openai-key

# For PDF processing (if adding new documents)
R2_ENDPOINT=your-r2-endpoint
R2_BUCKET=your-bucket
R2_ACCESS_KEY=your-access-key
R2_SECRET_KEY=your-secret-key
LLAMA_API_KEY=your-llama-parse-key

# Optional (for Cohere reranking - local reranker is default)
COHERE_API_KEY=your-cohere-key
```

### 3. Build Indices (First Time Only)

```bash
uv run python src/main.py
```

This creates:
- `chroma_db/` - Vector store with embeddings (24MB)
- `bm25_index.pkl` - BM25 keyword index (2.7MB)
- `chunks_output.json` - All chunks with metadata (1.8MB)

**Stats**: 1,877 chunks from 334 pages across 3 documents

### 4. Launch Streamlit App

```bash
uv run streamlit run streamlit/app.py
```

Open: http://localhost:8501

## 💬 Usage Examples

### Streamlit UI (Recommended)

1. Configure settings in sidebar:
   - ✅ Query Expansion (default: ON)
   - ✅ Reranking (default: ON, uses local model)
   - 📊 Top K candidates: 30
   - 📋 Final results: 10

2. Ask questions:
   - "How many annual leave days in IIMA?"
   - "Compare maternity leave policies across all organizations"
   - "What is the probation period in Chemexcil?"

3. View results:
   - Answer with inline citations
   - Confidence level (High/Medium/Low)
   - Retrieved chunks with scores
   - Full text preview

### Python API

```python
from retrieval import search

# Simple query
results = search(
    query="How many annual leave days in IIMA?",
    top_k=30,                    # Candidates after RRF
    rerank=True,                 # Enable reranking
    top_n=10,                    # Final results
    use_query_expansion=True,    # Query translation
    k_per_query=5,               # Results per retriever
    rerank_method="local"        # FREE local reranker
)

# Access results
for chunk in results.chunks:
    print(f"[{chunk.doc_id}] {chunk.section_title}")
    print(f"Score: {chunk.rerank_score:.3f}")
    print(f"Pages: {chunk.page_start}-{chunk.page_end}")
```

### With Answer Generation

```python
from retrieval import search
from answer_gen import generate_answer
from model.schema import FinalRankedResults, RankedChunk

# Retrieve chunks
results = search(query="Compare maternity leave policies")

# Convert to format for answer generation
ranked_chunks = [
    RankedChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,  # Full text
        chunk_summary=chunk.chunk_summary,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        rrf_score=chunk.rerank_score,
        appearances=1,
        sources=["hybrid"]
    )
    for chunk in results.chunks
]

final_ranked = FinalRankedResults(
    chunks=ranked_chunks,
    total_before_dedup=len(results.chunks),
    total_after_dedup=len(results.chunks)
)

# Generate answer
answer = generate_answer(query, final_ranked)
print(answer.answer)
print(f"Citations: {len(answer.citations)}")
print(f"Confidence: {answer.confidence}")
```

## 📂 Project Structure

```
Multi-Doc_RAG/
├── src/
│   ├── model/
│   │   └── schema.py          # Pydantic models
│   ├── r2/
│   │   ├── r2_client.py       # Cloudflare R2 operations
│   │   └── execute_for_r2.py  # PDF upload & parsing
│   ├── chunking.py            # Document chunking
│   ├── embed_store.py         # ChromaDB operations
│   ├── bm25_index.py          # BM25 index creation
│   ├── query_translate.py     # Query expansion
│   ├── retrieval.py           # Hybrid search & reranking
│   ├── answer_gen.py          # LLM answer generation
│   └── main.py                # Index building pipeline
├── streamlit/
│   ├── app.py                 # Streamlit UI
│   └── README.md              # UI documentation
├── doc/                       # Source PDF files
├── chroma_db/                 # Vector store (generated)
├── bm25_index.pkl             # BM25 index (generated)
├── chunks_output.json         # All chunks (generated)
├── .env                       # API keys (create this)
├── .gitignore                 # Git ignore rules
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

## 🎛️ Configuration

### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 30 | Candidates after RRF fusion |
| `top_n` | 10 | Final results after reranking |
| `k_per_query` | 5 | Results per retriever per query variant |
| `use_query_expansion` | True | Generate query variations |
| `rerank` | True | Enable reranking |
| `rerank_method` | "local" | "local" (free) or "cohere" (paid) |

### Reranking Options

#### Local Reranker (Default, FREE)
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Cost**: FREE
- **Speed**: ~9s for 30 candidates (after first load)
- **Quality**: ⭐⭐⭐⭐
- **Privacy**: Data stays local

#### Cohere Reranker (Optional, Paid)
- **Model**: `rerank-english-v3.0`
- **Cost**: $1 per 1000 searches
- **Speed**: ~500ms for 30 candidates
- **Quality**: ⭐⭐⭐⭐⭐
- **Privacy**: Data sent to Cohere API

### Performance

| Operation | Time (First Run) | Time (Cached) |
|-----------|------------------|---------------|
| Load indices | ~2-3s | ~200ms |
| Query expansion | ~1-2s | ~1-2s |
| Hybrid search (4 queries) | ~13s | ~1s |
| Local reranking (30→10) | ~64s (first) | ~9s |
| Answer generation | ~2-3s | ~2-3s |
| **Total** | **~80s** | **~13s** |

## 📚 Available Documents

- **IIMA** - Indian Institute of Management Ahmedabad
  - 1,148 chunks, 208 pages
  
- **Chemexcil** - Basic Chemicals, Cosmetics & Dyes Export Promotion Council
  - 320 chunks, 64 pages
  
- **TCCAP** - TCCAP HR Policies
  - 409 chunks, 62 pages

**Total**: 1,877 chunks, 334 pages

## 🔧 Advanced Usage

### Filtered Search

```python
from retrieval import search_with_filter

# Search specific documents only
results = search_with_filter(
    query="What is the maternity leave policy?",
    doc_ids=["iima", "chemexcil"],  # Only these docs
    top_k=10,
    rerank=True,
    top_n=5
)
```

### Without Query Expansion (Faster)

```python
results = search(
    query="Your question",
    use_query_expansion=False,  # Single query only
    top_k=20,                    # Adjust accordingly
    k_per_query=20               # More results per retriever
)
```

### Alternative Reranker Models

Edit `src/retrieval.py` line 390 to use different models:

```python
# Faster (but slightly lower quality)
CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')  # ~60MB

# Better quality (but slower)
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')  # ~120MB

# Best quality (slowest)
CrossEncoder('cross-encoder/ms-marco-electra-base')  # ~440MB
```

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
uv sync  # Install all dependencies
```

### "ChromaDB not found"
```bash
uv run python src/main.py  # Rebuild indices
```

### "Local reranker is slow"
- Normal on first run (~64s) while downloading model
- Subsequent runs are faster (~9s)
- Model is cached after first download

### "Out of memory"
- Reduce `top_k` parameter (try 20 instead of 30)
- Disable query expansion temporarily
- Use smaller reranker model

## 🚢 Production Deployment

### Option 1: Cloud Run (GCP)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

# Pre-download reranker model
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
```

### Option 2: Streamlit Cloud

1. Push repo to GitHub
2. Connect to Streamlit Cloud
3. Add environment variables (secrets)
4. Deploy!

### Optimization Tips

1. **Pre-build indices**: Include `chroma_db/` and `bm25_index.pkl` in deployment
2. **Cache model**: Download reranker model during build (not runtime)
3. **Use Cloud Storage**: Store indices in GCS/S3 for Cloud Run
4. **Set min instances**: Keep 1 instance warm to avoid cold starts
5. **Monitor performance**: Track query times and adjust parameters

## 📖 Documentation

- **Streamlit UI**: See `streamlit/README.md`
- **Reranker Comparison**: See `RERANKER_COMPARISON.md`
- **API Reference**: Docstrings in source files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- **LangChain**: Hybrid retrieval and document processing
- **ChromaDB**: Vector storage
- **Sentence Transformers**: Local cross-encoder reranking
- **OpenAI**: Embeddings and LLM generation
- **Streamlit**: Beautiful UI framework
- **LlamaParse**: PDF parsing

## 📧 Contact

[Add your contact information]

---

**Built with ❤️ for efficient multi-document RAG**

