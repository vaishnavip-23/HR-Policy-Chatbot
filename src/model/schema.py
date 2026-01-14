from pydantic import BaseModel, Field
from typing import List

class Chunk(BaseModel):
    # Identity
    chunk_id: str = Field(..., description="Unique ID: {doc_id}_{index}")
    doc_id: str = Field(..., description="chemexcil, iima, tccap")
    
    # Content
    text: str = Field(..., description="Full chunk text")
    chunk_summary: str = Field(..., description="2-3 sentence summary for embedding")
    
    # Filtering (CRITICAL for comparisons)
    section_title: str = Field(..., description="Section heading")
    
    # Citations
    page_start: int = Field(..., description="First page")
    page_end: int = Field(..., description="Last page")
    
    # Offsets (for mapping back)
    start_offset: int = Field(..., description="Char offset in full doc")
    end_offset: int = Field(..., description="Char offset in full doc")



class RetrievedChunk(BaseModel):
    """A chunk retrieved from hybrid search with relevance score."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Source document")
    text: str = Field(..., description="Full chunk text")
    chunk_summary: str = Field(..., description="Chunk summary")
    section_title: str = Field(..., description="Section heading")
    page_start: int = Field(..., description="First page")
    page_end: int = Field(..., description="Last page")
    relevance_score: float = Field(..., description="Relevance score from retriever")
    source: str = Field(..., description="Retrieval source: bm25, vector, or both")


class RerankedChunk(BaseModel):
    """A chunk after Cohere reranking with updated score."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Source document")
    text: str = Field(..., description="Full chunk text")
    chunk_summary: str = Field(..., description="Chunk summary")
    section_title: str = Field(..., description="Section heading")
    page_start: int = Field(..., description="First page")
    page_end: int = Field(..., description="Last page")
    rerank_score: float = Field(..., description="Cohere rerank relevance score (0-1)")
    original_score: float = Field(..., description="Original retrieval score")


class RetrievalResults(BaseModel):
    """Complete retrieval results with metadata."""
    query: str = Field(..., description="User query")
    chunks: List[RetrievedChunk] = Field(..., description="Retrieved chunks before reranking")
    total_candidates: int = Field(..., description="Total chunks before deduplication")
    unique_chunks: int = Field(..., description="Unique chunks after deduplication")


class FinalResults(BaseModel):
    """Final results after reranking."""
    query: str = Field(..., description="User query")
    chunks: List[RerankedChunk] = Field(..., description="Top chunks after reranking")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    rerank_time_ms: float = Field(..., description="Reranking time in milliseconds")



class InputQuery(BaseModel):
    query: str = Field(..., description="User query")


class QueryVariations(BaseModel):
    variations: List[str] = Field(..., min_length=3, max_length=3, description="3 query variations")


class FinalQueries(BaseModel):
    original_query: str = Field(..., description="Original user query")
    variations: List[str] = Field(..., min_length=3, max_length=3, description="3 query variations")



class RankedChunk(BaseModel):
    """A chunk after RRF ranking (used by alternative retrieval approach)."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Source document")
    text: str = Field(..., description="Full chunk text")
    chunk_summary: str = Field(..., description="Chunk summary")
    section_title: str = Field(..., description="Section heading")
    page_start: int = Field(..., description="First page")
    page_end: int = Field(..., description="Last page")
    rrf_score: float = Field(..., description="RRF score")
    appearances: int = Field(..., description="Number of times chunk appeared in retrieval")
    sources: List[str] = Field(..., description="Sources that retrieved this chunk")


class FinalRankedResults(BaseModel):
    """Final results after RRF merge and deduplication."""
    chunks: List[RankedChunk] = Field(..., description="Top ranked chunks after RRF")
    total_before_dedup: int = Field(..., description="Total chunks before deduplication")
    total_after_dedup: int = Field(..., description="Total unique chunks after deduplication")


class Citation(BaseModel):
    """Individual citation reference."""
    chunk_id: str = Field(..., description="Chunk ID used for this citation")
    doc_id: str = Field(..., description="Source document")
    section_title: str = Field(..., description="Section heading")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")


class Answer(BaseModel):
    """Final answer with citations."""
    answer: str = Field(..., description="Comprehensive answer to the user's query")
    citations: List[Citation] = Field(..., description="Citations used in the answer")
    confidence: str = Field(..., description="Confidence level: high, medium, or low")