import streamlit as st

from graph import app
from indexing import _format_metadata, _get_doc_name
from nodes import MAX_REWRITES, MissingAPIKeyError
from utils import load_documents


def get_doc_field(doc, field, default="N/A"):
    """Safely extract a field from a dict or LangChain Document."""
    if not doc:
        return default

    if isinstance(doc, dict):
        if doc.get(field) is not None:
            return doc[field]

        metadata = doc.get("metadata")
        if isinstance(metadata, dict):
            return metadata.get(field, default)
        return default

    if field == "content":
        return getattr(doc, "page_content", default) or default

    metadata = getattr(doc, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(field, default)
    return default


# 1. Page Configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
)

# 2. Header & System Capabilities
st.title("Agentic RAG AI Assistant")
st.markdown(
    """
This system performs:
- **Multi-Document Retrieval**
- **Retrieval Evaluation**
- **Query Rewriting**
- **Agentic Reasoning**
- **Grounded Answer Generation**
"""
)


# 3. Load Knowledge Base Documents
@st.cache_data(show_spinner=False)
def get_documents():
    return load_documents()


try:
    documents = get_documents()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("📄 View Loaded Knowledge Base Documents"):
    if not documents:
        st.info("No PDF files were found in the `data/` folder.")
    for doc in documents:
        st.write(f"**Document:** {_get_doc_name(doc)}")
        st.caption(_format_metadata(doc))
        st.divider()

# 4. User Query Input
question = st.text_input(
    "Ask your question",
    placeholder="e.g., What are the key skills listed in this resume?",
)

# 5. Execute Agentic RAG Workflow
if st.button("Run Agentic RAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner("Running Agentic RAG workflow..."):
            result = app.invoke(
                {
                    "query": question,
                    "question": question,
                    "current_query": question,
                    "documents": documents,
                    "rewrite_count": 0,
                }
            )
    except MissingAPIKeyError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # surface LLM / API failures instead of a traceback
        st.error(f"Workflow failed: {exc}")
        st.stop()

    if not documents:
        st.warning("No documents are loaded, so the answer is not grounded in a knowledge base.")

    rewrite_count = int(result.get("rewrite_count") or 0)
    if rewrite_count:
        st.subheader("🔄 Query Improvement")
        st.markdown(
            f"**Original Query**\n\n{question}\n\n"
            f"**Rewritten Query ({rewrite_count}/{MAX_REWRITES})**\n\n"
            f"{result.get('rewritten_question') or result.get('current_query', '')}"
        )

    grade = str(result.get("retrieval_grade", "")).lower()
    if grade == "yes":
        st.success("Retrieval Evaluation Passed")
    else:
        st.warning("Retrieval Evaluation Failed — the answer may be limited.")

    st.subheader("✨ Final AI Answer")
    st.info(result.get("answer") or "No answer generated.")

    st.subheader("🧠 Retrieval Reasoning")
    st.code(
        result.get("retrieval_reasoning") or "No reasoning available.",
        language="text",
    )

    with st.expander("📚 Retrieved Source Documents & Metadata"):
        retrieved_docs = result.get("retrieved_documents") or []
        if not retrieved_docs:
            st.write("No documents were retrieved.")
        for idx, doc in enumerate(retrieved_docs, start=1):
            st.markdown(f"**{idx}. Source:** `{_get_doc_name(doc)}`")
            st.caption(_format_metadata(doc))
            content = get_doc_field(doc, "content", "")
            if content:
                st.text(str(content)[:500] + "...")
            st.divider()
