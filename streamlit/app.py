import streamlit as st
import sys
from pathlib import Path

# Add project src/ to path dynamically (works in local and container runs)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from retrieval import search
from answer_gen import generate_answer
from model.schema import FinalRankedResults, RankedChunk

# Warmup system on first load (cached across reruns)
@st.cache_resource
def warmup_on_startup():
    """Load models and warm caches on app startup - runs only once."""
    from warmup import warmup_system
    warmup_system()

# Execute warmup
warmup_on_startup()

# Page config
st.set_page_config(
    page_title="HR Policy RAG Assistant",
    page_icon="📋",
    layout="wide"
)

# Title
st.title("📋 Multi-Doc HR Policy RAG Assistant")
st.markdown("Ask questions about HR policies (IIMA, Chemexcil, TCCAP) and get accurate answers with citations.")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Search Configuration")
    
    use_expansion = st.checkbox(
        "🔄 Query Expansion", 
        value=True,
        help="Generate 3 query variations for better recall"
    )
    
    use_reranking = st.checkbox(
        "🎯 Reranking", 
        value=True,
        help="Rerank results with cross-encoder for better relevance"
    )
    
    if use_reranking:
        rerank_method = st.radio(
            "Reranking Method",
            options=["local", "cohere"],
            index=0,
            help="Local: FREE cross-encoder (ms-marco) | Cohere: Paid API (better quality)"
        )
    else:
        rerank_method = "local"
    
    top_k = st.slider(
        "Candidates after RRF",
        min_value=10,
        max_value=50,
        value=35,  # ✅ Increased default
        step=5,
        help="Number of chunks after Reciprocal Rank Fusion"
    )
    
    top_n = st.slider(
        "Final results",
        min_value=3,
        max_value=15,
        value=10,
        step=1,
        help="Number of chunks to use for answer generation"
    )
    
    k_per_query = st.slider(
        "Results per retriever per query",
        min_value=3,
        max_value=10,
        value=6,  # ✅ Increased for better coverage
        step=1,
        help="How many results each retriever returns per query variant"
    )
    
    st.divider()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("📚 View Citations"):
                for i, citation in enumerate(message["citations"], 1):
                    st.markdown(f"**{i}.** Chunk `{citation['chunk_id']}` (Pages {citation['page_start']}-{citation['page_end']})")
            if "confidence" in message:
                confidence_emoji = {
                    "high": "🟢",
                    "medium": "🟡",
                    "low": "🔴"
                }
                st.caption(f"{confidence_emoji.get(message['confidence'], '⚪')} Confidence: **{message['confidence'].upper()}**")
            if "retrieval_time" in message and "rerank_time" in message:
                total_time = message["retrieval_time"] + message["rerank_time"]
                st.caption(f"⏱️ Response time: {total_time:.0f}ms")

# Chat input
if query := st.chat_input("Ask a question about HR policies..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            with st.status("Running pipeline...", expanded=True) as status:
                # Step 1: Retrieval
                if use_expansion:
                    status.write("🔄 Generating query variations...")
                    status.write("🔍 Running hybrid search (BM25 + Vector) on all variations...")
                else:
                    status.write("🔍 Running hybrid search (BM25 + Vector)...")
                
                # Call the actual search function with full text chunks
                results = search(
                    query=query,
                    top_k=top_k,
                    rerank=use_reranking,
                    top_n=top_n,
                    use_query_expansion=use_expansion,
                    k_per_query=k_per_query,
                    rerank_method=rerank_method
                )
                
                if use_reranking:
                    if rerank_method == "cohere":
                        status.write("🎯 Reranking with Cohere API...")
                    else:
                        status.write("🎯 Reranking with local cross-encoder...")
                else:
                    status.write("🔄 Fusing results with RRF...")
                
                # Convert FinalResults to FinalRankedResults for answer generation
                # IMPORTANT: Using chunk.text (full text), NOT chunk.chunk_summary
                ranked_chunks = []
                for chunk in results.chunks:
                    ranked_chunk = RankedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,  # ✅ FULL TEXT for context
                        chunk_summary=chunk.chunk_summary,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        rrf_score=chunk.rerank_score,
                        appearances=1,
                        sources=["hybrid"]
                    )
                    ranked_chunks.append(ranked_chunk)
                
                final_ranked = FinalRankedResults(
                    chunks=ranked_chunks,
                    total_before_dedup=top_k,
                    total_after_dedup=len(results.chunks)
                )
                
                status.write("💡 Generating answer with citations...")
                answer = generate_answer(query, final_ranked)
                status.update(label="✅ Done!", state="complete")
            
            # Display answer (includes inline citations)
            st.markdown(answer.answer)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer.answer,
                "citations": [
                    {
                        "chunk_id": c.chunk_id,
                        "page_start": c.page_start,
                        "page_end": c.page_end
                    } for c in answer.citations
                ],
                "confidence": answer.confidence,
                "retrieval_time": results.retrieval_time_ms,
                "rerank_time": results.rerank_time_ms
            })
            
            # Display citations
            with st.expander("📚 View Citations & Sources"):
                for i, citation in enumerate(answer.citations, 1):
                    st.markdown(f"**{i}.** Chunk `{citation.chunk_id}` (Pages {citation.page_start}-{citation.page_end})")
            
            # Confidence indicator
            confidence_emoji = {
                "high": "🟢",
                "medium": "🟡",
                "low": "🔴"
            }
            st.caption(f"{confidence_emoji.get(answer.confidence, '⚪')} Confidence: **{answer.confidence.upper()}**")
            
            # Display retrieval stats in expander
            with st.expander("📊 Retrieval Details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("📥 Candidates", top_k)
                col2.metric("📋 Final Chunks", len(results.chunks))
                col3.metric("⏱️ Total Time", f"{results.retrieval_time_ms + results.rerank_time_ms:.0f}ms")
                
                st.divider()
                st.caption("**Retrieved Chunks:**")
                for i, chunk in enumerate(results.chunks, 1):
                    with st.container():
                        st.markdown(f"**{i}. [{chunk.doc_id.upper()}] {chunk.section_title}**")
                        st.caption(f"Chunk ID: `{chunk.chunk_id}` | Pages {chunk.page_start}-{chunk.page_end} | Score: {chunk.rerank_score:.4f}")
                        with st.expander(f"View chunk {i} text"):
                            st.text(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            import traceback
            st.code(traceback.format_exc())

# Sidebar - About section
with st.sidebar:
    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system uses:
    - **Query Translation**: Generates 3 query variations (optional)
    - **Dense Retrieval**: Semantic search on chunk summaries
    - **Sparse Retrieval**: BM25 keyword search on full text
    - **RRF Fusion**: Reciprocal Rank Fusion for merging results
    - **Local Reranker**: FREE cross-encoder (ms-marco) for reranking
    - **LLM Generation**: GPT-5-mini with citations using **full text context**
    
    💡 **No reranking API costs!** Uses local sentence-transformers.
    """)
    
    st.divider()
    
    st.header("📚 Available Documents")
    st.markdown("""
    - **IIMA** - 1,148 chunks (208 pages)
    - **Chemexcil** - 320 chunks (64 pages)
    - **TCCAP** - 409 chunks (62 pages)
    
    **Total: 1,877 chunks**
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()