# src/llm_chain.py

from typing import Any, List, Dict
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

MODEL_NAME = "gemini-2.5-flash"
EXPECTED_METADATA_KEYS = ["merchant", "amount", "date"]

def build_chain_only(llm: Any = None):
    """
    Build a chain that combines given documents (with metadata) and answers a question.
    Does NOT perform retrieval itself.
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial assistant. Use the transaction metadata (merchant, amount, date). Compute sums and totals when asked."),
        ("human", """
Here are the transactions:
{context}

Question:
{input}
""")
    ])

    # document_prompt input_variables must match metadata keys + page_content
    document_prompt = PromptTemplate(
        input_variables=["page_content"] + EXPECTED_METADATA_KEYS,
        template=(
            "Merchant: {merchant}\n"
            "Amount: {amount}\n"
            "Date: {date}\n"
            "Detail: {page_content}"
        )
    )

    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=chat_prompt,
        document_prompt=document_prompt
    )

    return combine_chain

def answer_with_docs(chain: Any, docs: List[Document], question: str) -> str:
    """
    Invoke the chain by passing `context`=docs and `input`=question.
    Before invoking, ensure every Document.metadata has the required keys.
    """
    if chain is None:
        raise ValueError("Chain cannot be None")

    # Validate metadata for each document
    for idx, doc in enumerate(docs):
        md = doc.metadata or {}
        missing = [k for k in EXPECTED_METADATA_KEYS if k not in md]
        if missing:
            raise ValueError(f"Document at index {idx} is missing metadata keys {missing}")

    # Now invoke the chain
    result: Any = chain.invoke({"context": docs, "input": question})

    # Handle result being str or dict
    if isinstance(result, dict):
        return result.get("answer", str(result))
    # If result is string or other, just return it directly
    return str(result)
