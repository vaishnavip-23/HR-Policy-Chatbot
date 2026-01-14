from r2.r2_client import download_parsed_files
from chunking import chunking_markdown
from embed_store import embed_and_store
from bm25_index import create_bm25_index
import asyncio
import json




def print_chunk_details(chunk, show_full_text=False):
    """Print chunk details in a readable format."""
    print(f"\n{'─'*70}")
    print(f"📄 Chunk ID: {chunk.chunk_id}")
    print(f"📁 Document: {chunk.doc_id}")
    print(f"📍 Section: {chunk.section_title}")
    print(f"📖 Pages: {chunk.page_start}-{chunk.page_end}")
    print(f"📊 Offsets: {chunk.start_offset} → {chunk.end_offset}")
    print(f"\n💡 Summary:")
    print(f"   {chunk.chunk_summary}")
    
    if show_full_text:
        print(f"\n📝 Full Text (first 300 chars):")
        text_preview = chunk.text[:300] + ("..." if len(chunk.text) > 300 else "")
        print(f"   {text_preview}")
    
    print(f"{'─'*70}")


def print_chunk_samples(all_chunks, num_samples=3):
    """Print sample chunks from each document."""
    print(f"\n{'='*70}")
    print(f"SAMPLE CHUNKS (showing {num_samples} per document)")
    print(f"{'='*70}")
    
    # Group by doc_id
    chunks_by_doc = {}
    for chunk in all_chunks:
        if chunk.doc_id not in chunks_by_doc:
            chunks_by_doc[chunk.doc_id] = []
        chunks_by_doc[chunk.doc_id].append(chunk)
    
    # Print samples from each doc
    for doc_id, chunks in chunks_by_doc.items():
        print(f"\n📚 Document: {doc_id.upper()} ({len(chunks)} total chunks)")
        
        # Show first few chunks
        for i, chunk in enumerate(chunks[:num_samples]):
            print_chunk_details(chunk, show_full_text=True)


def save_chunks_to_json(all_chunks, filename="chunks_output.json"):
    """Save all chunks to JSON file for inspection."""
    chunks_data = [chunk.model_dump() for chunk in all_chunks]
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved all chunks to: {filename}")


async def main():
    """Process all HR policy documents and create chunks."""
    doc_ids = ["chemexcil", "iima", "tccap"]
    
    all_chunks = []
    
    for doc_id in doc_ids:
        print(f"\n{'='*60}")
        print(f"Processing document: {doc_id}")
        print(f"{'='*60}")
        
        # Download parsed files from R2
        markdown_text, page_map = download_parsed_files(doc_id)
        print(f"Downloaded markdown: {len(markdown_text)} characters")
        print(f"Page map: {len(page_map)} pages")
        
        # Chunk with section-aware splitting and summary generation
        chunks = await chunking_markdown(doc_id, markdown_text, page_map)
        all_chunks.extend(chunks)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Documents processed: {len(doc_ids)}")
    
    # Print sample chunks
    print_chunk_samples(all_chunks, num_samples=3)
    
    # Save all chunks to JSON
    save_chunks_to_json(all_chunks)
    
    # Create embeddings and store in ChromaDB
    print(f"\n{'='*60}")
    print(f"EMBEDDING & STORING IN VECTOR DB")
    print(f"{'='*60}")
    embed_and_store(all_chunks)
    
    # Create BM25 index
    print(f"\n{'='*60}")
    print(f"CREATING BM25 INDEX")
    print(f"{'='*60}")
    create_bm25_index(all_chunks)
    
    print(f"\n{'='*60}")
    print(f"✓ PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"Created:")
    print(f"  - {len(all_chunks)} chunks")
    print(f"  - ChromaDB vector store (./chroma_db/)")
    print(f"  - BM25 index (./bm25_index.pkl)")
    print(f"  - chunks_output.json")
    
    print(f"\nReady for retrieval!")
    print(f"  Run: uv run streamlit run streamlit/app.py")
    
    return all_chunks


if __name__ == "__main__":
    chunks = asyncio.run(main())

