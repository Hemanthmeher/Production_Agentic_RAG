import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from prompts import GRADE_PROMPT, RETRIEVAL_PROMPT, ANSWER_PROMPT

load_dotenv()

# ==============================================================================
# LLM
# ==============================================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ==============================================================================
# 1. RETRIEVAL NODE
# ==============================================================================


def retrieve_documents(state):
    query = state["query"]
    documents = state["documents"]

    document_list = ""
    for doc in documents:
        document_list += f"""
Document Name: {doc['document_name']}
Metadata:
{doc['metadata']}
-------------------------
"""

    prompt = RETRIEVAL_PROMPT.format(
        query=query,
        documents=document_list,
    )

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    reasoning = response.content

    selected_docs = []
    for doc in documents:
        if doc["document_name"] in reasoning:
            selected_docs.append(doc)

    return {
        "retrieved_docs": selected_docs,
        "retrieval_reasoning": reasoning,
    }


# ==============================================================================
# 2. GRADE DOCUMENTS NODE
# ==============================================================================


def grade_documents(state):
    query = state["current_query"]
    retrieved_docs = state["retrieved_docs"]

    if not retrieved_docs:
        return {"is_relevant": "no"}

    context = ""
    for doc in retrieved_docs:
        context += f"""
DOCUMENT:
{doc['document_name']}

CONTENT:
{doc['content'][:3000]}
"""

    prompt = GRADE_PROMPT.format(
        query=query,
        documents=context,
    )

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    grade = response.content.strip().lower()
    is_relevant = "yes" if "yes" in grade else "no"

    return {
        "is_relevant": is_relevant,
    }

# REWRITE QUERY
# ==============================================================================

from prompts import REWRITE_PROMPT


def rewrite_query(state):

    query = state["current_query"]

    prompt = REWRITE_PROMPT.format(
        query=query
    )

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    rewritten_query = (
        response.content
        .strip()
    )

    return {
        "current_query":"rewritten_query",
        "rewritten_query":rewritten_query
    }


def generate_answer(state):
    query = state["current_query"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            "answer": "I could not find sufficient information in the provided documents to answer this question."
        }

    context = ""
    for doc in retrieved_docs:
        doc_name = doc.get("document_name", "Unknown Document")
        content = doc.get("content", "")
        context += f"""
DOCUMENT:
{doc_name}

CONTENT:
{content}
-------------------------
"""

    prompt = ANSWER_PROMPT.format(
        query=query,
        context=context,
    )

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "answer": response.content.strip(),
    }