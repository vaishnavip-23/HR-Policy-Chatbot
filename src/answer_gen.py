import os
from openai import OpenAI
import instructor
from model.schema import FinalRankedResults, Answer, Citation
from dotenv import load_dotenv

load_dotenv()

# Initialize Instructor client with OpenAI Responses API
client = instructor.from_provider(
    "openai/gpt-5-mini",
    mode=instructor.Mode.RESPONSES_TOOLS
)


def generate_answer(query: str, final_results: FinalRankedResults) -> Answer:
    """
    Generate a comprehensive answer using retrieved chunks.
    
    Args:
        query: The original user query
        final_results: Top ranked chunks after RRF
        
    Returns:
        Answer object with answer text, citations, and confidence
    """
    print(f"\nGenerating answer for query: {query}")
    print(f"Using {len(final_results.chunks)} chunks")
    
    # Prepare context from chunks
    context_parts = []
    for i, chunk in enumerate(final_results.chunks, 1):
        context_parts.append(
            f"[Chunk {chunk.chunk_id}] (Pages {chunk.page_start}-{chunk.page_end})\n{chunk.text}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are an expert HR policy analyst with deep experience comparing organizational policies across multiple companies.

CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:

1. **Content Requirements:**
   - **IMPORTANT:** Check instructions 5 & 6 first for ambiguous or out-of-context queries
   - Provide a comprehensive, well-structured answer based ONLY on the provided context
   - For comparisons, create clear side-by-side summaries highlighting key differences
   - For complex questions requiring reasoning:
     * First summarize relevant facts from each organization
     * Then provide analysis or comparison with clear reasoning
     * Explicitly state if additional context would strengthen the answer
   - Use specific details and exact language from the context where possible

2. **Citation Format:**
   - Add inline citations immediately after each claim: [Chunk <chunk_id>, p.<page_start>-<page_end>]
   - Every factual statement must have a citation
   - For comparisons, cite each organization's policy separately

3. **Structured Formatting:**
   - Use bullet points for lists of items or policies
   - Use numbered lists for step-by-step processes
   - For multi-organization comparisons, use clear section headers or structured format
   - If one organization has more details than others, state that explicitly

4. **Confidence Scoring (IMPORTANT):**
   Set confidence level based on these strict criteria:
   
   - "high": 
     * Query is directly and completely answered with explicit facts
     * Multiple supporting citations (2+) from relevant chunks
     * No interpretation, inference, or speculation needed
     * Information is unambiguous and comprehensive
     * For comparisons: clear data from ALL requested organizations
   
   - "medium": 
     * Answer provided but requires some interpretation
     * Limited citations (1-2) or information from only some sources
     * Some details missing but core question is addressed
     * For comparisons: data from some but not all organizations
     * Reasoning or analysis required beyond direct facts
   
   - "low": 
     * Minimal relevant information found in context
     * Heavy interpretation or speculation required
     * Query not directly addressed in the chunks
     * For comparisons: only one organization has relevant data

5. **Ambiguous Queries:**
   If the query is too vague or ambiguous (e.g., "What are the leave policies?" without specifying which organization):
   - Ask a clarifying question
   - Suggest specific options (e.g., "Which organization? IIMA, CHEMEXCIL, or TCCAP?")
   - Or ask if they want a comparison across all organizations
   - Set confidence to "low"
   - Do NOT provide a full answer without clarification

6. **Out-of-Context Queries:**
   If the query is completely unrelated to the HR policy documents:
   - Keep response very brief (1-2 sentences maximum)
   - Simply state: "This question is outside the scope of the HR policy documents. I can only answer questions about IIMA, CHEMEXCIL, and TCCAP HR policies."
   - Do NOT include any citations
   - Set confidence to "low"
   - Do NOT attempt to answer the question generally

Return your answer in the structured format with proper citations and confidence level."""

    # Generate structured answer using instructor
    answer = client.responses.create(
        input=prompt,
        response_model=Answer
    )
    
    print(f"Answer generated with {len(answer.citations)} citations")
    print(f"Confidence: {answer.confidence}")
    
    return answer