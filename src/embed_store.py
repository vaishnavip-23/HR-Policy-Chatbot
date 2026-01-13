from typing import List
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from model.schema import Chunk

load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB (local, persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="vector_store",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)


def embed_and_store(chunks: List[Chunk]):
    """
    Embed chunk summaries and store in ChromaDB.
    - Embeddings: Created from chunk_summary (focused, semantic)
    - Documents: Full text (for LLM context)
    - Metadata: doc_id, section_title, page numbers, etc.
    """
    print(f"\nEmbedding and storing {len(chunks)} chunks...")
    
    # Extract summaries for embedding (dense retrieval)
    summaries = [chunk.chunk_summary for chunk in chunks]
    
    # Create embeddings using OpenAI on summaries
    print(f"  Generating embeddings...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=summaries
    )
    embeddings = [item.embedding for item in response.data]
    
    # Prepare data for ChromaDB
    ids = []
    documents = []  # Full text for retrieval results
    metadatas = []
    
    for chunk in chunks:
        ids.append(str(chunk.chunk_id))  # Chroma requires string IDs
        documents.append(chunk.text)  # Full text goes here
        metadatas.append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,  # CRITICAL for filtering
            "section_title": chunk.section_title,  # Context
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "chunk_summary": chunk.chunk_summary
        })
    
    # Store in ChromaDB
    print(f"  Storing in ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,  # From summaries (semantic search)
        documents=documents,     # Full text (returned in results)
        metadatas=metadatas     # Auxiliary info (filtering, citations)
    )
    
    print(f"✓ Stored {len(chunks)} chunks in vector_store")
    print(f"  - Embeddings: from chunk_summary")
    print(f"  - Documents: full chunk text")
    print(f"  - Metadata: doc_id, section_title, pages")
