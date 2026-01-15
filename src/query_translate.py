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
        input=f"""You are an expert at generating search query variations for an HR policy document retrieval system.

Generate 3 alternative queries that express the same intent using different wording and phrasing.

**HR POLICY CONTEXT:**
- The system contains HR policies from three organizations: IIMA, CHEMEXCIL, and TCCAP
- Each organization may use different terminology for the same HR concepts
- Company names may appear as: "IIMA", "Indian Institute of Management Ahmedabad", "CHEMEXCIL", "Basic Chemicals Cosmetics & Dyes Export Promotion Council", "TCCAP", "Tri-County Community Action Program"

**HR TERMINOLOGY VARIATIONS:**
When generating variations, consider that organizations may use different terms for the same concept:
- "Probation period" may also be called: "introductory period", "trial period", "probationary period", "provisional period"
- "Leave policy" may also be: "time off policy", "absence policy", "vacation policy"
- "Compensation" may also be: "salary", "pay", "remuneration", "benefits"
- "Termination" may also be: "separation", "exit", "dismissal", "resignation"
- "Performance review" may also be: "appraisal", "evaluation", "assessment"
- "Notice period" may also be: "resignation period", "separation notice"
- "Annual leave" may also be: "vacation days", "paid time off", "PTO"
- "Sick leave" may also be: "medical leave", "illness leave"
- "Maternity leave" may also be: "parental leave", "pregnancy leave"

**COMPANY NAME AWARENESS:**
- If the query mentions specific company names (IIMA, CHEMEXCIL, TCCAP), PRESERVE them in your variations
- Keep company names explicit to help the system retrieve relevant chunks from those specific organizations
- If comparing multiple companies, maintain all company names in the variations
- If asking about "all organizations" or no specific company is mentioned, keep it general

**RULES:**
1. Preserve the original meaning and intent
2. Maintain any company names mentioned in the original query
3. Incorporate HR terminology variations where appropriate
4. Each variation should be a standalone search query
5. Do NOT introduce new facts or answer the query
6. Focus on terminology that would help retrieve relevant chunks from different organizations

**EXAMPLES:**

Original: "What is the probation period in IIMA?"
Good variations:
- "IIMA introductory period duration for new employees"
- "How long is the trial period at Indian Institute of Management Ahmedabad?"
- "IIMA probationary period policy details"

Original: "Compare leave policies between IIMA and TCCAP"
Good variations:
- "Differences in time off policies between IIMA and TCCAP"
- "IIMA vs TCCAP vacation and absence policy comparison"
- "Leave entitlements at IIMA compared to TCCAP"

Original: "What is the maternity leave policy?"
Good variations:
- "Parental leave policy for pregnancy"
- "Maternity and pregnancy leave entitlements"
- "Medical leave policy for expecting mothers"

User query: {query}

Generate 3 alternative query variations following the guidelines above.""",
        response_model=QueryVariations,
    )
    
    print(f"Generated {len(response.variations)} variations")
    
    # Return FinalQueries with original + variations
    return FinalQueries(original_query=query, variations=response.variations)