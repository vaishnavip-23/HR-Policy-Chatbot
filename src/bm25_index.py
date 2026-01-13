from typing import List
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from model.schema import Chunk

# Path
BM25_INDEX_PATH = "./bm25_index.pkl"


def create_bm25_index(chunks: List[Chunk]):
    """
    Create BM25 index from chunks and save to disk.
    Uses LangChain's BM25Retriever for keyword search.
    """
    print(f"\nCreating BM25 index from {len(chunks)} chunks...")
    
    # Convert chunks to LangChain Document format
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.text,  # Full text for keyword search
            metadata={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section_title": chunk.section_title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_summary": chunk.chunk_summary,
            }
        )
        documents.append(doc)
    
    # Create BM25 retriever
    print(f"  Building BM25 inverted index...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10  # Default top-k
    
    # Save to disk
    print(f"  Saving to {BM25_INDEX_PATH}...")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    
    print(f"✓ BM25 index saved to {BM25_INDEX_PATH}")
    print(f"  - Indexed {len(chunks)} chunks")
    print(f"  - Keyword search ready")


def load_bm25_index():
    """Load pre-built BM25 index from disk."""
    print(f"Loading BM25 index from {BM25_INDEX_PATH}...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    print(f"✓ BM25 index loaded")
    return bm25_retriever

