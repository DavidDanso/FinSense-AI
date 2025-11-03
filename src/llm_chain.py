from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import Any, Dict

MODEL_NAME = "gemini-2.5-flash"

def build_qa_chain(retriever, llm: Any = None) -> Any:
    """
    Builds and returns a retrieval + LLM chain.
    retriever: a retriever instance (from your retrieval logic)
    llm: optional chat model; if not provided, uses Google Gemini model
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial assistant with 25 years of Professional experience. Use the transaction data context to answer user questions clearly and accurately."),
        ("human", """
Here are the transactions:
{context}

Question:
{question}
""")
    ])

    # Use combine_documents chain to stuff retrieved docs into prompt
    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=chat_prompt
    )

    # Create retrieval chain that uses retriever + combine_chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_chain
    )

    return retrieval_chain

def answer_question(chain: Any, question: str) -> str:
    """
    Invoke the chain on a question. Returns narrative answer.
    """
    result: Dict = chain.invoke({"input": question})
    answer = result.get("answer")
    return answer
