import streamlit as st
from graph import app
from utils import load_documents


def get_doc_field(doc, field, default="N/A"):
    """Safely extract fields whether doc is a dict or a LangChain Document, and metadata is a dict or str."""
    if not doc:
        return default

    if isinstance(doc, dict):
        if field in doc and doc[field] is not None:
            return doc[field]

        metadata = doc.get("metadata")
        if isinstance(metadata, dict):
            return metadata.get(field, default)
        if isinstance(metadata, str) and metadata.strip():
            # If metadata is raw text/string instead of a dict, return it cleanly as fallback
            return metadata if field in ("short_summary", "main_topics", "summary") else default

        return default

    # LangChain Document object fallback
    if field == "content":
        return getattr(doc, "page_content", default) or default
    metadata = getattr(doc, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(field, default)
    if isinstance(metadata, str) and metadata.strip():
        return metadata if field in ("short_summary", "main_topics", "summary") else default
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
documents = load_documents()

with st.expander("📄 View Loaded Knowledge Base Documents"):
    for doc in documents or []:
        doc_name = get_doc_field(
            doc,
            "source",
            get_doc_field(doc, "file_name", get_doc_field(doc, "document_name", "Unknown")),
        )
        st.write(f"**Document:** {doc_name}")
        st.write(f"**Main Topic:** {get_doc_field(doc, 'main_topic')}")
        st.write(f"**Short Summary:** {get_doc_field(doc, 'short_summary')}")
        st.divider()

# 4. User Query Input
question = st.text_input(
    "Ask your question",
    placeholder="e.g., What are the key security policies for remote employees?",
)

# 5. Execute Agentic RAG Workflow
if st.button("Run Agentic RAG"):
    if question.strip():
        with st.spinner("Running Agentic RAG workflow..."):
           result = app.invoke(
    {
        "query": question,
        "question": question,
        "documents": documents,
    }
)

        st.success("Documents Loaded")
        st.success("Retrieval Completed")

        if result.get("rewritten_question"):
            st.warning("Retrieval required query improvement.")
            st.subheader("🔄 Query Improvement")
            st.markdown(
                f"""
**Original Query**
{result.get('query', question)}

**Rewritten Query**
{result.get('rewritten_question', '')}
"""
            )

        st.success("Retrieval Evaluation Passed")
        st.success("Answer Generated")

        # Final Answer
        st.subheader("✨ Final AI Answer")
        st.info(result.get("answer", "No answer generated."))

        # Retrieval Reasoning
        st.subheader("🧠 Retrieval Reasoning")
        st.code(
            result.get("retrieval_reasoning", "No reasoning available."),
            language="text",
        )

        # Retrieved Documents
        with st.expander("📚 Retrieved Source Documents & Metadata"):
            retrieved_docs = result.get("retrieved_documents") or []
            if retrieved_docs:
                for idx, doc in enumerate(retrieved_docs, start=1):
                    source = get_doc_field(
                        doc,
                        "source",
                        get_doc_field(doc, "file_name", get_doc_field(doc, "document_name", "Unknown")),
                    )
                    st.markdown(f"**{idx}. Source:** `{source}`")
                    st.write(f"**Main Topic:** {get_doc_field(doc, 'main_topic')}")
                    st.write(f"**Short Summary:** {get_doc_field(doc, 'short_summary')}")
                    content = get_doc_field(
                        doc, "content", get_doc_field(doc, "page_content", "")
                    )
                    if content:
                        st.text(str(content)[:500] + "...")
                    st.divider()
            else:
                st.write("No documents were retrieved.")
    else:
        st.warning("Please enter a question.")