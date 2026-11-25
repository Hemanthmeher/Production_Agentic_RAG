import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

def generate_document_metadata(document):
    prompt = f"""
Analyze this document.

Return:

1. Short Summary
2. Main Topics

Keep concise.

Document:
{document['content'][:4000]}
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return response.content

