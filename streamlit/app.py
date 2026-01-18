import streamlit as st
import sys
from pathlib import Path

# Add project src/ to path dynamically (works in local and container runs)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from retrieval import search
from answer_gen import generate_answer
from model.schema import FinalRankedResults, RankedChunk
from safety_guard import is_query_safe
from query_classifier import classify_query, IntentType

# Page config
st.set_page_config(
    page_title="HR Policy RAG Assistant",
    page_icon="📋",
    layout="wide"
)

# Title
st.title("📋 HR Policy RAG Assistant")
st.markdown("Ask questions about HR policies of organizations IIMA, Chemexcil, TCCAP and get accurate answers with citations.")

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
                    st.markdown(f"**{i}.** Chunk {citation['chunk_id']} (Pages {citation['page_start']}-{citation['page_end']})")
            if "confidence" in message:
                confidence_color = {
                    "high": "🟢",
                    "medium": "🟡",
                    "low": "🔴"
                }
                st.caption(f"{confidence_color.get(message['confidence'], '⚪')} Confidence: {message['confidence']}")

# Chat input
if query := st.chat_input("Ask a question about HR policies..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Step 1: Safety Check
            safety_placeholder = st.empty()
            safety_placeholder.info("🛡️ Checking query safety...")
            
            is_safe, safety_reason = is_query_safe(query, use_llm_check=True)
            
            if not is_safe:
                # Query is unsafe - display warning and block
                safety_placeholder.empty()
                st.error(safety_reason)
                st.warning("If you believe this is an error, please rephrase your question.")
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"🚫 {safety_reason}\n\nPlease ask a legitimate question about HR policies."
                })
                st.stop()
            
            safety_placeholder.success("✅ Query is safe")
            
            # Step 2: Intent Classification
            intent_placeholder = st.empty()
            intent_placeholder.info("🎯 Classifying query intent...")
            
            intent, fixed_response = classify_query(query, use_llm=True)
            
            if intent != IntentType.HR_POLICY_QUESTION:
                # Handle with fixed response (no RAG needed)
                intent_placeholder.success(f"✅ Intent: {intent.value}")
                st.markdown(fixed_response)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fixed_response
                })
                st.stop()
            
            intent_placeholder.success("✅ Intent: HR Policy Question")
            
            # Clear status messages before showing answer
            safety_placeholder.empty()
            intent_placeholder.empty()
            
            # Step 3: RAG Pipeline (only for HR policy questions)
            with st.status("Running RAG pipeline...", expanded=True) as status:
                status.write("Translating query and running hybrid search (BM25 + Vector)...")
                results = search(query)
                
                status.write("Reranking with local cross-encoder...")
                
                status.write("Generating answer with inline citations...")
                
                # Convert to FinalRankedResults for answer generation
                ranked_chunks = [
                    RankedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        chunk_summary=chunk.chunk_summary,
                        section_title=chunk.section_title,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        rrf_score=chunk.rerank_score,
                        appearances=1,
                        sources=["hybrid"]
                    )
                    for chunk in results.chunks
                ]
                
                final_results = FinalRankedResults(
                    chunks=ranked_chunks,
                    total_before_dedup=len(results.chunks),
                    total_after_dedup=len(results.chunks)
                )
                
                answer = generate_answer(query, final_results)
                status.update(label="Done!", state="complete")
            
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
                "confidence": answer.confidence
            })
            
            # Display citations
            with st.expander("📚 View Citations"):
                for i, citation in enumerate(answer.citations, 1):
                    st.markdown(f"**{i}.** Chunk {citation.chunk_id} (Pages {citation.page_start}-{citation.page_end})")
            
            # Display retrieval stats and confidence in sidebar
            with st.sidebar:
                st.subheader("📊 Retrieval Stats")
                st.metric("Chunks Retrieved", final_results.total_before_dedup)
                st.metric("Unique Chunks", final_results.total_after_dedup)
                st.metric("Top Chunks Used", len(final_results.chunks))
                st.caption(f"Confidence: {answer.confidence}")
                
                with st.expander("🔍 View Retrieved Chunks"):
                    for i, chunk in enumerate(final_results.chunks, 1):
                        sources_str = " + ".join(chunk.sources)
                        st.markdown(f"**{i}. Chunk {chunk.chunk_id}**")
                        st.caption(f"RRF Score: {chunk.rrf_score:.4f} | Sources: {sources_str} | Pages: {chunk.page_start}-{chunk.page_end}")
                        st.text(chunk.text[:200] + "...")
                        st.divider()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system uses:
    - **🛡️ Safety Guard**: Detects and blocks prompt injection attempts
    - **🎯 Intent Classification**: Fast responses for greetings and meta-questions
    - **Query Translation**: Generates multiple query variations
    - **Dense Retrieval**: Semantic search on chunk summaries
    - **Sparse Retrieval**: BM25 keyword search on full text
    - **RRF Reranking**: Reciprocal Rank Fusion for merging results
    - **LLM Answer Generation**: Answers using GPT-5-mini along with citations
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
