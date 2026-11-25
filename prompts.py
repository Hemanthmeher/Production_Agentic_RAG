# ==============================================================================
# RETRIEVAL PROMPT
# ==============================================================================

RETRIEVAL_PROMPT = """
You are an intelligent retrieval assistant.

User Question:
{query}

Available Documents:
{documents}

Your task:
1. Analyze the question carefully.
2. Identify the MOST relevant documents.
3. Select the top relevant documents.
4. Explain WHY they are relevant.

Return format:

Document:
Reason:
"""

# ==============================================================================
# DOCUMENT GRADING PROMPT
# ==============================================================================

GRADE_PROMPT = """
You are a document relevance evaluator.

User Question:
{query}

Retrieved Content:
{documents}

Determine whether the retrieved content
contains enough information to answer the question.

Return ONLY:

yes

or

no
"""


# ==============================================================================
# QUERY REWRITE PROMPT
# ==============================================================================

REWRITE_PROMPT = """
You are an expert query rewriting system.

Original Question:
{query}

The retrieved documents were insufficient.

Rewrite the question to improve retrieval quality.

Return ONLY the rewritten question.
"""


# ==============================================================================
# ANSWER GENERATION PROMPT
# ==============================================================================

ANSWER_PROMPT = """
You are an advanced enterprise AI assistant.

Answer ONLY using the provided context.

Question:
{query}

Context:
{context}

Requirements:

- accurate
- concise
- grounded
- professional
- no hallucinations

If information is unavailable,
say that it is not present in the documents.
"""