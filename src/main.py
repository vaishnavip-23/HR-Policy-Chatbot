from r2.r2_client import download_parsed_files
from chunking import chunking_markdown
from embed_store import embed_and_store
from bm25_index import create_bm25_index
import asyncio

all_chunks = []

for doc_id in ["chemexcil", "iima", "tccap"]:
    markdown_text, page_map = download_parsed_files(doc_id)
    chunks = asyncio.run(chunking_markdown(doc_id, markdown_text, page_map))
    all_chunks.extend(chunks)

embed_and_store(all_chunks)
create_bm25_index(all_chunks)
