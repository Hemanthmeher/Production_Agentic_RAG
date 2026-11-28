from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # Query tracking
    query: str
    question: str
    current_query: str
    rewritten_question: str
    rewrite_count: int

    # Knowledge base / retrieval
    documents: list[dict[str, Any]]
    retrieved_documents: list[dict[str, Any]]
    retrieval_reasoning: str
    retrieval_grade: str  # "yes" | "no"

    # Output
    answer: str
