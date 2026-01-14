"""System warmup to eliminate cold starts."""

def warmup_system():
    """
    Pre-load all models and indices.
    Call this once at server startup to eliminate first-query latency.
    """
    print("🔥 Warming up Multi-Doc RAG system...")
    
    # 1. Load reranker model
    print("  📥 Loading reranker model...")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("  ✓ Reranker model loaded")
    
    # 2. Load retrievers (ChromaDB + BM25)
    print("  📥 Loading retrievers...")
    from retrieval import load_retrievers
    load_retrievers()
    print("  ✓ ChromaDB and BM25 retrievers loaded")
    
    # 3. Run a test query to warm all caches
    print("  🧪 Running warmup query...")
    from retrieval import search
    search(
        query="warmup test query",
        top_k=10,
        rerank=True,
        top_n=3,
        use_query_expansion=False,
        k_per_query=5,
        rerank_method="local"
    )
    print("  ✓ System cache warmed")
    
    print("✅ System ready! First query will now be fast.")


if __name__ == "__main__":
    warmup_system()

