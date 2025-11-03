# llm_chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import Any

MODEL_NAME = "gemini-2.5-flash"

def build_qa_chain(retriever, llm=None) -> Any:
    """
    Builds and returns a retrieval + LLM chain.
    retriever: a retriever instance (from your RetrieverService)
    llm: optional ChatOpenAI or other chat model
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    # Build the chat prompt template with roles
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial assistant. Use the transaction data context to answer user questions clearly and accurately."),
        ("human", """
    Here are the transactions:
    {context}

    Question:
    {question}
    """)
    ])

    combine_chain = create_stuff_documents_chain(llm=ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0), prompt=chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

    return retrieval_chain

def answer_question(chain, question: str) -> str:
    """
    Invoke the chain on a question. Returns narrative answer.
    """
    result = chain.invoke({"input": question})
    answer = result.get("answer")
    return answer
