from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import Any, Dict

MODEL_NAME = "gemini-2.5-flash"

def build_qa_chain(retriever, llm: Any = None) -> Any:
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial assistant. Use transaction data (merchant, amount, date) to answer questions. When the question asks totals or sums, compute using the amount metadata."),
        ("human", """
Here are the transaction records:
{context}

Question:
{input}
""")
    ])

    document_prompt = PromptTemplate(
        input_variables=["page_content", "metadata"],
        template=(
            "Merchant: {metadata[merchant]}\n"
            "Amount: {metadata[amount]}\n"
            "Date: {metadata[date]}\n"
            "Note: {page_content}"
        )
    )

    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=chat_prompt,
        document_prompt=document_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_chain
    )

    return retrieval_chain

def answer_question(chain: Any, question: str) -> str:
    result: Dict = chain.invoke({"input": question})
    return result.get("answer", "No answer returned.")
