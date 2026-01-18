"""
Query Intent Classification Module
Classifies user queries to determine how to handle them
"""
import re
import instructor
from enum import Enum
from dotenv import load_dotenv
from model.schema import IntentClassificationResult

load_dotenv()

# Initialize Instructor client with OpenAI Responses API
client = instructor.from_provider(
    "openai/gpt-5-mini",
    mode=instructor.Mode.RESPONSES_TOOLS
)


class IntentType(Enum):
    """Types of user intents"""
    GREETING = "greeting"
    ABOUT_APP = "about_app"
    HR_POLICY_QUESTION = "hr_policy_question"
    OUT_OF_SCOPE = "out_of_scope"


# Predefined responses for different intents
INTENT_RESPONSES = {
    IntentType.GREETING: """Hello! 👋 I'm your HR Policy Assistant. I can help you find information about HR policies from IIMA, Chemexcil, and TCCAP organizations.

Feel free to ask me questions like:
- "What is the leave policy?"
- "How do I apply for maternity leave?"
- "What are the working hours?"

How can I assist you today?""",
    
    IntentType.ABOUT_APP: """I'm an AI-powered HR Policy Assistant designed to help you find accurate information from multiple HR policy documents.

**What I can do:**
- Answer questions about HR policies from IIMA, Chemexcil, and TCCAP
- Provide answers with direct citations from the source documents
- Handle complex queries about leave, compensation, benefits, working hours, and more

**How I work:**
- I use advanced retrieval techniques (hybrid search with BM25 + vector embeddings)
- I rerank results for accuracy using cross-encoder models
- I generate answers with inline citations so you can verify the information

**What I can't do:**
- I only have access to HR policies for IIMA, Chemexcil, and TCCAP
- I can't answer general questions outside of these policies
- I can't make decisions or provide legal advice

Ask me anything about the HR policies I have access to!""",
    
    IntentType.OUT_OF_SCOPE: """I'm sorry, but I can only help with questions related to HR policies from IIMA, Chemexcil, and TCCAP organizations.

**Topics I can help with:**
- Leave policies (sick leave, casual leave, maternity/paternity leave)
- Working hours and attendance
- Compensation and benefits
- Employee grievances and conduct
- Performance reviews
- And more...

Please ask a question related to these HR policies, and I'll be happy to help!"""
}


# Rule-based patterns for quick classification
GREETING_PATTERNS = [
    r"^(hi|hello|hey|good morning|good afternoon|good evening|greetings|howdy)[\s\.,!?]*$",
    r"^(hi|hello|hey)\s+(there|friend|assistant)[\s\.,!?]*$",
    r"^how are you[\s\.,!?]*$",
    r"^what'?s up[\s\.,!?]*$",
    r"^yo[\s\.,!?]*$",
]

ABOUT_APP_PATTERNS = [
    r"what (can|do) you do",
    r"how (do|does) (you|this|it) work",
    r"tell me about (yourself|this app|this system)",
    r"what (is|are) (your|you) (capabilities|features)",
    r"what (kind of|type of) questions can (i|you) ask",
    r"help me",
    r"what (are|do) you know",
    r"explain (yourself|what you do)",
]


def rule_based_classification(query: str) -> IntentType | None:
    """
    Fast rule-based classification for obvious cases.
    
    Returns:
        IntentType if confident, None if uncertain
    """
    query_lower = query.lower().strip()
    
    # Check for greetings
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, query_lower):
            return IntentType.GREETING
    
    # Check for about app questions
    for pattern in ABOUT_APP_PATTERNS:
        if re.search(pattern, query_lower):
            return IntentType.ABOUT_APP
    
    # Check for obvious HR policy keywords
    hr_keywords = [
        "leave", "salary", "policy", "policies", "hr", "employee",
        "vacation", "sick", "maternity", "paternity", "compensation",
        "benefits", "working hours", "attendance", "grievance",
        "performance", "appraisal", "termination", "resignation",
        "probation", "notice period", "iima", "chemexcil", "tccap"
    ]
    
    if any(keyword in query_lower for keyword in hr_keywords):
        return IntentType.HR_POLICY_QUESTION
    
    # If uncertain, return None (will fall back to LLM)
    return None


def llm_classification(query: str) -> IntentType:
    """
    Use LLM for sophisticated intent classification.
    
    Returns:
        IntentType
    """
    try:
        prompt = f"""You are an intent classifier for an HR Policy RAG system.

Classify queries into one of these categories:
1. "greeting" - Simple greetings or casual conversation starters
2. "about_app" - Questions about what the system can do, its capabilities, or how it works
3. "hr_policy_question" - Questions about HR policies from IIMA, Chemexcil, or TCCAP organizations
4. "out_of_scope" - Questions outside of HR policies (weather, general knowledge, etc.)

Classify this query: {query}

Return the intent category and confidence score (0.0 to 1.0)."""

        result = client.responses.create(
            input=prompt,
            response_model=IntentClassificationResult
        )
        
        # Map string to enum
        intent_map = {
            "greeting": IntentType.GREETING,
            "about_app": IntentType.ABOUT_APP,
            "hr_policy_question": IntentType.HR_POLICY_QUESTION,
            "out_of_scope": IntentType.OUT_OF_SCOPE
        }
        
        return intent_map.get(result.intent, IntentType.HR_POLICY_QUESTION)
    
    except Exception as e:
        print(f"LLM classification error: {e}")
        # Default to HR policy question if classification fails
        return IntentType.HR_POLICY_QUESTION


def classify_query(query: str, use_llm: bool = True) -> tuple[IntentType, str | None]:
    """
    Main classification function that combines rule-based and LLM approaches.
    
    Args:
        query: User query to classify
        use_llm: Whether to use LLM for uncertain cases (default: True)
    
    Returns:
        tuple: (intent_type, fixed_response_or_none)
        - If intent is not HR_POLICY_QUESTION, returns a fixed response
        - If intent is HR_POLICY_QUESTION, returns None (proceed to RAG)
    """
    print(f"\n{'='*60}")
    print(f"🎯 INTENT CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Query: {query}")
    
    # First try rule-based classification
    print(f"  Step 1: Rule-based classification...")
    intent = rule_based_classification(query)
    
    if intent is not None:
        print(f"  ✓ Detected by rules: {intent.value}")
    else:
        print(f"  → Uncertain, needs LLM classification")
    
    # If uncertain and LLM is available, use it
    if intent is None and use_llm:
        print(f"  Step 2: LLM-based classification...")
        intent = llm_classification(query)
        print(f"  ✓ LLM classified as: {intent.value}")
    
    # If still uncertain, default to HR policy question
    if intent is None:
        intent = IntentType.HR_POLICY_QUESTION
        print(f"  → Defaulting to: {intent.value}")
    
    # Return intent and fixed response (if any)
    if intent == IntentType.HR_POLICY_QUESTION:
        print(f"✓ Intent: {intent.value} → Proceeding to RAG pipeline")
        print(f"{'='*60}\n")
        return intent, None
    else:
        print(f"✓ Intent: {intent.value} → Using fixed response")
        print(f"{'='*60}\n")
        return intent, INTENT_RESPONSES[intent]
