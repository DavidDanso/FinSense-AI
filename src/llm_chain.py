# llm_chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import Any, Dict

MODEL_NAME = "gemini-2.5-flash"

def build_qa_chain(retriever, llm: Any = None) -> Any:
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    # Use “input” in the human template, because create_retrieval_chain uses input key
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial assistant. Use the transaction data context to answer user questions clearly and accurately."),
        ("human", """
Here are the relevant transactions:
{context}

Question:
{input}
""")
    ])

    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_chain
    )

    return retrieval_chain

def answer_question(chain: Any, question: str) -> str:
    # Pass the question under key "input"
    result: Dict = chain.invoke({"input": question})
    return result.get("answer", "No answer returned.")
