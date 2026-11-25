from typing import TypedDict


class AgentState(TypedDict):
    query: str
    current_query: str
    rewritten_query: str
    documents: list
    retrieved_docs: list
    retrieval_grade: str
    retrieval_reasoning: str
    answer: str