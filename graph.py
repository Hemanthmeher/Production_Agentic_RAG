from langgraph.graph import END, START, StateGraph

from State import AgentState
from nodes import (
    MAX_REWRITES,
    generate_answer,
    grade_documents,
    retrieve_documents,
    rewrite_query,
)


def route_after_grading(state: AgentState) -> str:
    """Send good retrievals to the answer, and weak ones to query rewriting."""
    grade = str(state.get("retrieval_grade", "no")).strip().lower()

    if grade.startswith("yes"):
        return "generate_answer"

    # Stop looping when the rewrite budget is exhausted.
    if int(state.get("rewrite_count") or 0) >= MAX_REWRITES:
        return "generate_answer"

    return "rewrite_query"


workflow = StateGraph(AgentState)

workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query",
    },
)

workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("generate_answer", END)

app = workflow.compile()
