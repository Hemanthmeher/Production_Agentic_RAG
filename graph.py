from langgraph.graph import StateGraph, START, END

from State import AgentState
from nodes import (
    retrieve_documents,
    grade_documents,
    rewrite_query,
    generate_answer,
)


def route_after_grading(state: AgentState):
    grade = state.get("retrieval_grade", "no").strip().lower()
    if grade == "yes":
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