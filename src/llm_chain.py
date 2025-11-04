# llm_chain.py

from typing import Any, List
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

MODEL_NAME = "gemini-2.5-flash"
EXPECTED_METADATA_KEYS = ["merchant", "amount", "date"]

def build_chain_only(llm: Any = None):
    """Build versatile chain that handles any question type."""
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are FinSense AI, a creative and intelligent financial assistant. "
         "You can handle ANY type of question about transaction data - analytical, creative, storytelling, or conversational. "
         "\nYour capabilities:"
         "\n- Calculate totals, averages, counts with precision"
         "\n- Tell engaging stories about spending patterns"
         "\n- Provide insights and observations"
         "\n- Answer creatively while staying accurate with the data"
         "\n- Adapt your response style to match the user's question tone"
         "\n\nIMPORTANT FORMATTING RULES:"
         "\n- Use markdown formatting for better readability"
         "\n- Use **bold** for merchant names and important amounts"
         "\n- Use line breaks between paragraphs"
         "\n- Use bullet points (•) for lists instead of long sentences"
         "\n- Keep paragraphs short (2-3 sentences max)"
         "\n- For stories: structure with clear paragraphs, not walls of text"
         "\n\nFor creative questions (stories, narratives): be engaging with clear paragraph breaks"
         "\nFor analytical questions (totals, counts): be precise and use bullet points for clarity"
         "\nFor exploratory questions: provide insights in digestible chunks"),
        ("human", """
Transaction Data:
{context}

User Question: {input}

Respond naturally and appropriately. Use markdown formatting for readability.
""")
    ])

    document_prompt = PromptTemplate(
        input_variables=["page_content"] + EXPECTED_METADATA_KEYS,
        template="Merchant: {merchant}, Amount: ${amount}, Date: {date}, Details: {page_content}"
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

    for idx, doc in enumerate(docs):
        md = doc.metadata or {}
        missing = [k for k in EXPECTED_METADATA_KEYS if k not in md]
        if missing:
            raise ValueError(f"Document at index {idx} is missing metadata keys {missing}")

    result: Any = chain.invoke({"context": docs, "input": question})

    if isinstance(result, dict):
        return result.get("answer", str(result))
    return str(result)