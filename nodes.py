import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from indexing import _format_metadata, _get_doc_name
from prompts import ANSWER_PROMPT, GRADE_PROMPT, RETRIEVAL_PROMPT, REWRITE_PROMPT

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_REWRITES = int(os.getenv("MAX_QUERY_REWRITES", "2"))

NO_CONTEXT_ANSWER = (
    "I could not find sufficient information in the provided documents to answer "
    "this question."
)


class MissingAPIKeyError(RuntimeError):
    """Raised when the Groq API key is not configured."""


# ==============================================================================
# LLM
# ==============================================================================


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Create the LLM lazily so importing this module never fails."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Add it to a .env file in the project root "
            "or export it before starting the app."
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        temperature=0,
    )


def _invoke(prompt: str) -> str:
    response = get_llm().invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    return (content or "").strip()


# ==============================================================================
# HELPERS
# ==============================================================================


def _doc_content(doc) -> str:
    if isinstance(doc, dict):
        return doc.get("content") or doc.get("page_content") or ""
    return getattr(doc, "page_content", "") or ""


def _current_query(state) -> str:
    return (
        state.get("current_query")
        or state.get("query")
        or state.get("question")
        or ""
    ).strip()


def _mentions_document(doc, text: str) -> bool:
    name = _get_doc_name(doc).strip().lower()
    if not name or name == "unknown":
        return False

    stem = os.path.splitext(name)[0]
    candidates = {name, stem, stem.replace("_", " "), stem.replace("-", " ")}
    return any(len(candidate) >= 4 and candidate in text for candidate in candidates)


# ==============================================================================
# 1. RETRIEVAL NODE
# ==============================================================================


def retrieve_documents(state) -> dict:
    query = _current_query(state)
    documents = state.get("documents") or []

    if not documents:
        return {
            "current_query": query,
            "retrieved_documents": [],
            "retrieval_reasoning": "No documents are loaded in the knowledge base.",
        }

    catalog = []
    for index, doc in enumerate(documents, start=1):
        catalog.append(
            f"Document {index}:\n"
            f"Document Name: {_get_doc_name(doc)}\n"
            f"Metadata: {_format_metadata(doc)}\n"
            f"Content Preview:\n{_doc_content(doc)[:1200]}\n"
        )

    prompt = RETRIEVAL_PROMPT.format(
        query=query,
        documents="\n---\n".join(catalog),
    )
    reasoning = _invoke(prompt)

    lowered = reasoning.lower()
    selected_docs = [doc for doc in documents if _mentions_document(doc, lowered)]

    # Never hand an empty set to the grader; fall back to the full knowledge base.
    if not selected_docs:
        selected_docs = documents

    return {
        "current_query": query,
        "retrieved_documents": selected_docs,
        "retrieval_reasoning": reasoning,
    }


# ==============================================================================
# 2. GRADE DOCUMENTS NODE
# ==============================================================================


def grade_documents(state) -> dict:
    query = _current_query(state)
    retrieved_docs = state.get("retrieved_documents") or []

    if not retrieved_docs:
        return {"retrieval_grade": "no"}

    context = "\n".join(
        f"DOCUMENT:\n{_get_doc_name(doc)}\n\nCONTENT:\n{_doc_content(doc)[:3000]}\n"
        for doc in retrieved_docs
    )

    prompt = GRADE_PROMPT.format(query=query, documents=context)
    grade = _invoke(prompt).strip().lower().strip(".:!*` \n")

    is_relevant = grade.startswith("yes")
    return {"retrieval_grade": "yes" if is_relevant else "no"}


# ==============================================================================
# 3. REWRITE QUERY NODE
# ==============================================================================


def rewrite_query(state) -> dict:
    query = _current_query(state)
    attempt = int(state.get("rewrite_count") or 0) + 1

    prompt = REWRITE_PROMPT.format(query=query)
    rewritten_query = _invoke(prompt).strip().strip('"').strip("'") or query

    return {
        "current_query": rewritten_query,
        "rewritten_question": rewritten_query,
        "rewrite_count": attempt,
    }


# ==============================================================================
# 4. ANSWER GENERATION NODE
# ==============================================================================


def generate_answer(state) -> dict:
    query = _current_query(state)
    retrieved_docs = state.get("retrieved_documents") or []

    if not retrieved_docs:
        return {"answer": NO_CONTEXT_ANSWER}

    context = "\n".join(
        f"DOCUMENT:\n{_get_doc_name(doc)}\n\nCONTENT:\n"
        f"{_doc_content(doc)[:4000]}\n-------------------------\n"
        for doc in retrieved_docs
    )

    prompt = ANSWER_PROMPT.format(query=query, context=context)
    return {"answer": _invoke(prompt) or NO_CONTEXT_ANSWER}
