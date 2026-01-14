from dotenv import load_dotenv
from model.schema import QueryVariations, FinalQueries, InputQuery
import instructor
import os

load_dotenv()

# Initialize Instructor with Responses API mode
client = instructor.from_provider(
    "openai/gpt-5-mini",
    mode=instructor.Mode.RESPONSES_TOOLS
)


def query_translate(query: str) -> FinalQueries:
    input_query = InputQuery(query=query)
    response = client.responses.create(
        input=f"""You are given a user query.

    Generate 3 alternative queries that express the same intent
    using different wording and phrasing.

    Rules:
    - Preserve the original meaning.
    - Do NOT introduce new facts.
    - Do NOT answer the query.
    - Each variation should be a standalone search query.
    - You can introduce new keywords which might help with the retrieval process.

    Return only the list of rewritten queries.

    User query: {query}""",
        response_model=QueryVariations,
    )
    
    print(f"Generated {len(response.variations)} variations")
    
    # Return FinalQueries with original + variations
    return FinalQueries(original_query=query, variations=response.variations)