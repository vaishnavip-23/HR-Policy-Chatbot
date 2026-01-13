from typing import List, Dict, Any
import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from model.schema import Chunk
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def chunking_markdown(doc_id: str, markdown_text: str, page_map: Dict[str, Any]) -> List[Chunk]:
    """
    Section-aware chunking with LLM-generated summaries.
    
    Args:
        doc_id: Document identifier (e.g., 'chemexcil', 'iima', 'tccap')
        markdown_text: Full markdown text from LlamaParse
        page_map: Page mapping with character offsets
    
    Returns:
        List of Chunk objects with metadata
    """
    print(f"\nChunking document: {doc_id}")
    
    # Step 1: Split by markdown headers (section-aware)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep headers in content for context
    )
    
    header_splits = markdown_splitter.split_text(markdown_text)
    print(f"  Split into {len(header_splits)} sections by headers")
    
    # Step 2: Further split long sections if needed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    
    all_docs = []
    for doc in header_splits:
        # If section is too long, split further
        if len(doc.page_content) > 1200:
            sub_docs = text_splitter.create_documents(
                texts=[doc.page_content],
                metadatas=[doc.metadata]
            )
            all_docs.extend(sub_docs)
        else:
            all_docs.append(doc)
    
    print(f"  After size-based splitting: {len(all_docs)} chunks")
    
    # Step 3: Generate summaries in parallel
    texts = [doc.page_content for doc in all_docs]
    print(f"  Generating summaries for {len(texts)} chunks...")
    summaries = await generate_summaries(texts)
    
    # Step 4: Create Chunk objects
    chunks = []
    for idx, (doc, summary) in enumerate(zip(all_docs, summaries), start=1):
        chunk_text = doc.page_content
        
        # Extract section title from markdown headers
        section_title = extract_section_title(doc.metadata)
        
        # Calculate character offsets in original markdown
        start_offset = find_chunk_offset(chunk_text, markdown_text)
        end_offset = start_offset + len(chunk_text) - 1
        
        # Map character offsets to page numbers
        page_start, page_end = get_pages_for_chunk(start_offset, end_offset, page_map)
        
        chunk = Chunk(
            chunk_id=f"{doc_id}_{idx}",
            doc_id=doc_id,
            text=chunk_text,
            chunk_summary=summary,
            section_title=section_title,
            start_offset=start_offset,
            end_offset=end_offset,
            page_start=page_start,
            page_end=page_end
        )
        chunks.append(chunk)
    
    print(f"✓ Created {len(chunks)} chunks for {doc_id}")
    return chunks


def extract_section_title(metadata: dict) -> str:
    """
    Extract section title from markdown header metadata.
    Priority: h3 > h2 > h1 > "General"
    """
    if "h3" in metadata and metadata["h3"]:
        return metadata["h3"]
    elif "h2" in metadata and metadata["h2"]:
        return metadata["h2"]
    elif "h1" in metadata and metadata["h1"]:
        return metadata["h1"]
    return "General"


def find_chunk_offset(chunk_text: str, full_text: str) -> int:
    """
    Find the character offset of chunk in the full markdown text.
    Uses first 100 characters for matching to handle minor variations.
    """
    # Try to find exact match first
    offset = full_text.find(chunk_text)
    if offset != -1:
        return offset
    
    # Fallback: match on first 100 chars (handles whitespace differences)
    search_text = chunk_text[:min(100, len(chunk_text))].strip()
    offset = full_text.find(search_text)
    if offset != -1:
        return offset
    
    # Last resort: return 0
    return 0


def get_pages_for_chunk(start_offset: int, end_offset: int, page_map: Dict[str, Any]) -> tuple:
    """
    Map character offsets to page numbers using page_map.
    
    Args:
        start_offset: Start character position in full document
        end_offset: End character position in full document
        page_map: Dictionary mapping page indices to offset ranges
    
    Returns:
        Tuple of (page_start, page_end)
    """
    page_start = None
    page_end = None
    
    for page_idx, page_info in page_map.items():
        page_start_off = page_info["start_offset"]
        page_end_off = page_info["end_offset"]
        
        # Check if chunk start falls within this page
        if page_start_off <= start_offset <= page_end_off:
            page_start = page_info["page"]
        
        # Check if chunk end falls within this page
        if page_start_off <= end_offset <= page_end_off:
            page_end = page_info["page"]
    
    # Fallback to first page if start not found
    if page_start is None:
        page_start = page_map["0"]["page"]
    
    # Fallback to last page if end not found
    if page_end is None:
        last_key = max(page_map.keys(), key=lambda x: int(x))
        page_end = page_map[last_key]["page"]
    
    return page_start, page_end


async def generate_summaries(texts: List[str]) -> List[str]:
    """
    Generate summaries for all chunks in parallel using OpenAI.
    """
    tasks = [summarize_single_text(text) for text in texts]
    summaries = await asyncio.gather(*tasks)
    return summaries


async def summarize_single_text(text: str) -> str:
    """
    Generate a concise summary for a single chunk using OpenAI Responses API.
    Focuses on key facts, numbers, and important conditions.
    """
    prompt = "Write 2-3 sentences summarizing the main information in this text, focusing on key facts, numbers, dates, and important conditions."
    
    # Limit text length to save tokens (first 800 chars usually capture key info)
    text_snippet = text[:800]
    input_text = f"{prompt}\n\nText: {text_snippet}"
    
    try:
        response = await openai_client.responses.create(
            model="gpt-5-mini",
            input=input_text
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"  Warning: Summary generation failed ({e}), using fallback")
        # Fallback: return first 200 chars + ellipsis
        return text[:200].strip() + "..."

