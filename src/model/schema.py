from pydantic import BaseModel, Field

class Chunk(BaseModel):
    # Identity
    chunk_id: str = Field(..., description="Unique ID: {doc_id}_{index}")
    doc_id: str = Field(..., description="chemexcil, iima, tccap")
    
    # Content
    text: str = Field(..., description="Full chunk text")
    chunk_summary: str = Field(..., description="2-3 sentence summary for embedding")
    
    # Filtering (CRITICAL for comparisons)
    hr_topic: str = Field(..., description="leave_policy, compensation, benefits, etc.")
    section_title: str = Field(..., description="Section heading")
    
    # Citations
    page_start: int = Field(..., description="First page")
    page_end: int = Field(..., description="Last page")
    
    # Offsets (for mapping back)
    start_offset: int = Field(..., description="Char offset in full doc")
    end_offset: int = Field(..., description="Char offset in full doc")