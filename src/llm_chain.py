# llm_chain.py

from typing import Any, List
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

MODEL_NAME = "gemini-2.5-flash"
EXPECTED_METADATA_KEYS = ["merchant", "amount", "date"]

def build_chain_only(llm: Any = None):
    """Build chain that processes documents with transaction metadata."""
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a financial assistant. You have access to transaction data. "
         "When calculating totals or aggregations, use ALL transactions provided. "
         "Be precise with calculations - sum all amounts carefully."),
        ("human", """
Transaction Data:
{context}

Question: {input}

Instructions: 
- For total/sum questions: Add up ALL transaction amounts provided
- For specific merchant questions: Filter and analyze relevant transactions
- Always show your calculation if computing totals
""")
    ])

    document_prompt = PromptTemplate(
        input_variables=["page_content"] + EXPECTED_METADATA_KEYS,
        template=(
            "Merchant: {merchant} | Amount: {amount} | Date: {date} | Detail: {page_content}\n"
        )
    )

    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=chat_prompt,
        document_prompt=document_prompt
    )

    return combine_chain

def answer_with_docs(chain: Any, docs: List[Document], question: str) -> str:
    """Invoke chain with documents and question."""
    if chain is None:
        raise ValueError("Chain cannot be None")

    # Validate metadata
    for idx, doc in enumerate(docs):
        md = doc.metadata or {}
        missing = [k for k in EXPECTED_METADATA_KEYS if k not in md]
        if missing:
            raise ValueError(f"Document at index {idx} is missing metadata keys {missing}")

    result: Any = chain.invoke({"context": docs, "input": question})

    if isinstance(result, dict):
        return result.get("answer", str(result))
    return str(result)