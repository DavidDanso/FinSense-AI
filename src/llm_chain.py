# llm_chain.py

from typing import Any, List
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

MODEL_NAME = "gemini-2.5-flash"
EXPECTED_METADATA_KEYS = ["merchant", "amount", "date"]

def build_chain_only(llm: Any = None):
    """Build chain with optimized prompt for concise financial answers."""
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are FinSense AI, a helpful financial assistant. "
         "Provide clear, concise, and accurate answers about transaction data. "
         "Rules: "
         "1. For totals/sums: State the final amount directly "
         "2. For counts: State the count directly "
         "3. For specific merchant queries: Summarize (total spent, number of transactions) "
         "4. Only list individual transactions if explicitly asked to 'list' or 'show' them "
         "5. Use natural, conversational language "
         "6. Be precise with numbers and calculations"),
        ("human", """
Transaction Data Available:
{context}

User Question: {input}

Provide a direct, concise answer. Calculate accurately and respond naturally.
""")
    ])

    document_prompt = PromptTemplate(
        input_variables=["page_content"] + EXPECTED_METADATA_KEYS,
        template="{merchant}|${amount}|{date}"
    )

    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=chat_prompt,
        document_prompt=document_prompt
    )

    return combine_chain

def answer_with_docs(chain: Any, docs: List[Document], question: str) -> str:
    """Invoke chain with validated documents and return answer."""
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