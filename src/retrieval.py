from typing import List, Optional, Dict
import pickle
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from model.schema import RetrievedChunk, RerankedChunk, RetrievalResults, FinalResults

load_dotenv()

# Global reranker model (loaded once)
_reranker_model = None

# Paths
CHROMADB_DIR = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index.pkl"

# Global retrievers (loaded once)
_vector_retriever = None
_bm25_retriever = None


def load_retrievers():
    """
    Load pre-built indices (ChromaDB + BM25).
    Called once at startup or first query.
    Returns tuple: (vector_retriever, bm25_retriever)
    """
    global _vector_retriever, _bm25_retriever
    
    if _vector_retriever is not None and _bm25_retriever is not None:
        return _vector_retriever, _bm25_retriever
    
    print("Loading retrievers...")
    
    # Load ChromaDB (vector search)
    print("  Loading ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma = Chroma(
        collection_name="vector_store",
        embedding_function=embeddings,
        persist_directory=CHROMADB_DIR
    )
    _vector_retriever = chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )
    
    # Load BM25 (keyword search)
    print("  Loading BM25 index...")
    with open(BM25_INDEX_PATH, "rb") as f:
        _bm25_retriever = pickle.load(f)
    _bm25_retriever.k = 20
    
    print("✓ Retrievers loaded")
    return _vector_retriever, _bm25_retriever


def reciprocal_rank_fusion(results_list: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Combine multiple retrieval results using Reciprocal Rank Fusion (RRF).
    
    Args:
        results_list: List of ranked document lists from different retrievers
        k: Constant for RRF formula (default: 60)
    
    Returns:
        Fused and re-ranked list of documents
    """
    # Calculate RRF scores
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.metadata.get("chunk_id", str(id(doc)))
            
            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            
            doc_scores[doc_id] += rrf_score
    
    # Sort by RRF score (descending)
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    # Return ranked documents
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]


def hybrid_search(query: str, top_k: int = 20, k_per_retriever: int = 10) -> RetrievalResults:
    """
    Perform hybrid search (BM25 + Vector) using Reciprocal Rank Fusion.
    
    Args:
        query: User query
        top_k: Number of candidates to retrieve after fusion (before reranking)
        k_per_retriever: Number of results to fetch from each retriever per query
    
    Returns:
        RetrievalResults with retrieved chunks
    """
    print(f"\n{'='*60}")
    print(f"Hybrid Search: {query}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load both retrievers
    vector_retriever, bm25_retriever = load_retrievers()
    
    # Update k for retrievers
    bm25_retriever.k = k_per_retriever
    
    # Retrieve from both sources
    print(f"  BM25 keyword search (k={k_per_retriever})...")
    bm25_results = bm25_retriever.invoke(query)
    print(f"    → {len(bm25_results)} results")
    
    print(f"  Vector semantic search (k={k_per_retriever})...")
    # For vector retriever, we need to recreate it with new k
    vector_retriever.search_kwargs["k"] = k_per_retriever
    vector_results = vector_retriever.invoke(query)
    print(f"    → {len(vector_results)} results")
    
    # Apply Reciprocal Rank Fusion
    print(f"  Applying RRF...")
    fused_results = reciprocal_rank_fusion([bm25_results, vector_results])
    
    # Take top_k after fusion
    fused_results = fused_results[:top_k]
    
    # Convert to RetrievedChunk objects
    chunks = []
    seen_chunk_ids = set()
    
    for rank, doc in enumerate(fused_results, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        
        # Deduplicate
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        
        # Determine source
        in_bm25 = any(d.metadata.get("chunk_id") == chunk_id for d in bm25_results)
        in_vector = any(d.metadata.get("chunk_id") == chunk_id for d in vector_results)
        
        if in_bm25 and in_vector:
            source = "both"
        elif in_bm25:
            source = "bm25"
        else:
            source = "vector"
        
        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=doc.metadata.get("doc_id"),
            text=doc.page_content,
            chunk_summary=doc.metadata.get("chunk_summary", ""),
            section_title=doc.metadata.get("section_title", ""),
            page_start=doc.metadata.get("page_start"),
            page_end=doc.metadata.get("page_end"),
            relevance_score=1.0 / rank,  # Use inverse rank as score
            source=source
        )
        chunks.append(chunk)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"✓ Retrieved {len(chunks)} unique chunks in {elapsed_ms:.1f}ms")
    print(f"  - {sum(1 for c in chunks if c.source == 'both')} from both retrievers")
    print(f"  - {sum(1 for c in chunks if c.source == 'bm25')} BM25 only")
    print(f"  - {sum(1 for c in chunks if c.source == 'vector')} Vector only")
    
    return RetrievalResults(
        query=query,
        chunks=chunks,
        total_candidates=len(bm25_results) + len(vector_results),
        unique_chunks=len(chunks)
    )


def hybrid_search_with_expansion(query: str, top_k: int = 30, k_per_query: int = 5) -> RetrievalResults:
    """
    Perform hybrid search with query expansion.
    Uses query_translate to generate variations, retrieves from each, and fuses results.
    
    Args:
        query: Original user query
        top_k: Number of candidates to retrieve after fusion (before reranking)
        k_per_query: Number of results per retriever per query variation (default: 5)
    
    Returns:
        RetrievalResults with retrieved chunks from all query variations
    """
    from query_translate import query_translate
    
    print(f"\n{'='*60}")
    print(f"Hybrid Search with Query Expansion")
    print(f"{'='*60}")
    print(f"Original query: {query}")
    
    start_time = time.time()
    
    # Step 1: Generate query variations
    print(f"\n📝 Generating query variations...")
    expanded_queries = query_translate(query)
    all_queries = [expanded_queries.original_query] + expanded_queries.variations
    
    print(f"\n🔍 Searching with {len(all_queries)} query variations:")
    for i, q in enumerate(all_queries, 1):
        label = "Original" if i == 1 else f"Variation {i-1}"
        print(f"   {i}. [{label}] {q}")
    
    # Step 2: Load retrievers
    vector_retriever, bm25_retriever = load_retrievers()
    
    # Step 3: Retrieve from all query variations
    all_bm25_results = []
    all_vector_results = []
    
    print(f"\n🔎 Retrieving {k_per_query} from each retriever per query...")
    for i, query_var in enumerate(all_queries, 1):
        print(f"\n   Query {i}/{len(all_queries)}: '{query_var[:60]}...'")
        
        # Update k
        bm25_retriever.k = k_per_query
        vector_retriever.search_kwargs["k"] = k_per_query
        
        # BM25 search
        bm25_results = bm25_retriever.invoke(query_var)
        all_bm25_results.extend(bm25_results)
        print(f"      BM25: {len(bm25_results)} results")
        
        # Vector search
        vector_results = vector_retriever.invoke(query_var)
        all_vector_results.extend(vector_results)
        print(f"      Vector: {len(vector_results)} results")
    
    print(f"\n📊 Total retrieved:")
    print(f"   BM25: {len(all_bm25_results)} results")
    print(f"   Vector: {len(all_vector_results)} results")
    print(f"   Combined: {len(all_bm25_results) + len(all_vector_results)} results")
    
    # Step 4: Apply RRF to fuse all results
    print(f"\n🔄 Applying Reciprocal Rank Fusion...")
    fused_results = reciprocal_rank_fusion([all_bm25_results, all_vector_results])
    
    # Take top_k after fusion
    fused_results = fused_results[:top_k]
    
    # Step 5: Convert to RetrievedChunk objects
    chunks = []
    seen_chunk_ids = set()
    
    for rank, doc in enumerate(fused_results, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        
        # Deduplicate
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        
        # Determine source
        in_bm25 = any(d.metadata.get("chunk_id") == chunk_id for d in all_bm25_results)
        in_vector = any(d.metadata.get("chunk_id") == chunk_id for d in all_vector_results)
        
        if in_bm25 and in_vector:
            source = "both"
        elif in_bm25:
            source = "bm25"
        else:
            source = "vector"
        
        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=doc.metadata.get("doc_id"),
            text=doc.page_content,
            chunk_summary=doc.metadata.get("chunk_summary", ""),
            section_title=doc.metadata.get("section_title", ""),
            page_start=doc.metadata.get("page_start"),
            page_end=doc.metadata.get("page_end"),
            relevance_score=1.0 / rank,  # Use inverse rank as score
            source=source
        )
        chunks.append(chunk)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"\n✓ Retrieved {len(chunks)} unique chunks in {elapsed_ms:.1f}ms")
    print(f"  - {sum(1 for c in chunks if c.source == 'both')} from both retrievers")
    print(f"  - {sum(1 for c in chunks if c.source == 'bm25')} BM25 only")
    print(f"  - {sum(1 for c in chunks if c.source == 'vector')} Vector only")
    
    return RetrievalResults(
        query=query,
        chunks=chunks,
        total_candidates=len(all_bm25_results) + len(all_vector_results),
        unique_chunks=len(chunks)
    )


def load_reranker_model():
    """
    Load local cross-encoder reranker model (one-time load).
    Uses sentence-transformers - free and runs locally!
    """
    global _reranker_model
    
    if _reranker_model is not None:
        return _reranker_model
    
    print("Loading local reranker model...")
    from sentence_transformers import CrossEncoder
    
    # Using MS MARCO cross-encoder - optimized for search relevance
    # Options (from fastest to most accurate):
    # - ms-marco-MiniLM-L-6-v2  (fastest, ~80MB)
    # - ms-marco-MiniLM-L-12-v2 (balanced, ~120MB)
    # - ms-marco-TinyBERT-L-6   (tiny, ~60MB)
    _reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("✓ Local reranker model loaded (ms-marco-MiniLM-L-6-v2)")
    return _reranker_model


def rerank_with_local_model(
    query: str,
    retrieval_results: RetrievalResults,
    top_n: int = 10
) -> FinalResults:
    """
    Rerank retrieved chunks using local cross-encoder model (FREE!).
    Uses sentence-transformers cross-encoder for semantic reranking.
    
    Args:
        query: User query
        retrieval_results: Results from hybrid_search
        top_n: Number of top chunks to return after reranking
    
    Returns:
        FinalResults with reranked chunks
    """
    print(f"\n🔄 Reranking {len(retrieval_results.chunks)} candidates with local model...")
    
    start_time = time.time()
    
    # Load model (cached after first call)
    model = load_reranker_model()
    
    # Prepare query-document pairs
    pairs = [[query, chunk.text] for chunk in retrieval_results.chunks]
    
    # Score all pairs (batch processing)
    print(f"  Computing relevance scores...")
    scores = model.predict(pairs)
    
    # Sort by score (descending)
    scored_chunks = list(zip(retrieval_results.chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_chunks = scored_chunks[:top_n]
    
    # Convert to RerankedChunk objects
    reranked_chunks = []
    for rank, (chunk, score) in enumerate(top_chunks, 1):
        reranked_chunk = RerankedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            chunk_summary=chunk.chunk_summary,
            section_title=chunk.section_title,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            rerank_score=float(score),  # Cross-encoder score
            original_score=chunk.relevance_score
        )
        reranked_chunks.append(reranked_chunk)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"✓ Reranked to top {len(reranked_chunks)} chunks in {elapsed_ms:.1f}ms")
    print(f"  Score range: {reranked_chunks[0].rerank_score:.3f} → {reranked_chunks[-1].rerank_score:.3f}")
    
    return FinalResults(
        query=query,
        chunks=reranked_chunks,
        retrieval_time_ms=0.0,  # Set by caller
        rerank_time_ms=elapsed_ms
    )


def search(
    query: str, 
    top_k: int = 35, 
    rerank: bool = True, 
    top_n: int = 15,  
    use_query_expansion: bool = True,  
    k_per_query: int = 6
) -> FinalResults:
    """
    Complete search pipeline: hybrid retrieval with query expansion + local reranking.
    
    Default behavior (recommended):
    - Expands query into 4 variants (1 original + 3 variations)
    - Retrieves 6 results from BM25 and 6 from Vector per query = 12 total per query
    - Total: 4 queries × 12 results = ~48 results (with duplicates)
    - After RRF fusion: top_k=35 unique results
    - After local rerank: top_n=15 final results 
    
    Args:
        query: User query
        top_k: Number of candidates after RRF fusion (default: 35)
        rerank: Whether to use local reranking (default: True)
        top_n: Number of final results after reranking (default: 15, increased for better multi-org coverage)
        use_query_expansion: Use query translation for better recall (default: True)
        k_per_query: Results per retriever per query variant (default: 6)
    
    Returns:
        FinalResults with best matching chunks
    """
    retrieval_start = time.time()
    
    # Step 1: Hybrid retrieval (with or without query expansion)
    if use_query_expansion:
        retrieval_results = hybrid_search_with_expansion(
            query, 
            top_k=top_k,
            k_per_query=k_per_query
        )
    else:
        # Standard hybrid search without expansion
        retrieval_results = hybrid_search(
            query, 
            top_k=top_k,
            k_per_retriever=k_per_query * 4  # Compensate for no expansion
        )
    
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    # Step 2: Local reranking
    if rerank and len(retrieval_results.chunks) > 0:
        final_results = rerank_with_local_model(query, retrieval_results, top_n=top_n)
        
        final_results.retrieval_time_ms = retrieval_time
        return final_results
    else:
        # No reranking: convert top chunks to RerankedChunk format
        reranked_chunks = [
            RerankedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_summary=chunk.chunk_summary,
                section_title=chunk.section_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                rerank_score=1.0,  # No reranking
                original_score=chunk.relevance_score
            )
            for chunk in retrieval_results.chunks[:top_n]
        ]
        
        return FinalResults(
            query=query,
            chunks=reranked_chunks,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=0.0
        )


def search_with_filter(
    query: str, 
    doc_ids: Optional[List[str]] = None,
    top_k: int = 20,
    rerank: bool = True,
    top_n: int = 5
) -> FinalResults:
    """
    Search with document filtering (e.g., only IIMA, or IIMA + CHEMEXCIL).
    
    Args:
        query: User query
        doc_ids: List of doc_ids to filter (None = all documents)
        top_k: Number of candidates from hybrid search
        rerank: Whether to use local reranking
        top_n: Number of final results
    
    Returns:
        FinalResults with filtered and ranked chunks
    """
    print(f"\n{'='*60}")
    print(f"Filtered Search: {query}")
    if doc_ids:
        print(f"Documents: {', '.join(doc_ids)}")
    print(f"{'='*60}")
    
    # For filtered search, we use ChromaDB directly with metadata filter
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma = Chroma(
        collection_name="vector_store",
        embedding_function=embeddings,
        persist_directory=CHROMADB_DIR
    )
    
    # Build filter
    search_kwargs = {"k": top_k}
    if doc_ids:
        if len(doc_ids) == 1:
            search_kwargs["filter"] = {"doc_id": doc_ids[0]}
        else:
            search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}
    
    retriever = chroma.as_retriever(search_kwargs=search_kwargs)
    
    retrieval_start = time.time()
    documents = retriever.invoke(query)
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    # Convert to RetrievalResults
    chunks = []
    for doc in documents:
        chunk = RetrievedChunk(
            chunk_id=doc.metadata.get("chunk_id"),
            doc_id=doc.metadata.get("doc_id"),
            text=doc.page_content,
            chunk_summary=doc.metadata.get("chunk_summary", ""),
            section_title=doc.metadata.get("section_title", ""),
            page_start=doc.metadata.get("page_start"),
            page_end=doc.metadata.get("page_end"),
            relevance_score=0.0,
            source="vector_filtered"
        )
        chunks.append(chunk)
    
    retrieval_results = RetrievalResults(
        query=query,
        chunks=chunks,
        total_candidates=len(chunks),
        unique_chunks=len(chunks)
    )
    
    print(f"✓ Retrieved {len(chunks)} filtered chunks")
    
    # Rerank if enabled
    if rerank and len(chunks) > 0:
        final_results = rerank_with_local_model(query, retrieval_results, top_n=top_n)
        final_results.retrieval_time_ms = retrieval_time
        return final_results
    else:
        reranked_chunks = [
            RerankedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_summary=chunk.chunk_summary,
                section_title=chunk.section_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                rerank_score=1.0,
                original_score=chunk.relevance_score
            )
            for chunk in chunks[:top_n]
        ]
        
        return FinalResults(
            query=query,
            chunks=reranked_chunks,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=0.0
        )



