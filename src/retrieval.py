# retrieval.py

from typing import List, Dict, Any
import pandas as pd
from langchain_core.documents import Document

class RetrieverService:
    def __init__(self, embedding_manager, **retriever_kwargs):
        """
        embedding_manager: your EmbeddingManager instance
        retriever_kwargs: options you pass to as_retriever (e.g. k, search_type, etc)
        """
        self.embedding_manager = embedding_manager
        self.retriever_kwargs = retriever_kwargs

    def get_retriever(self):
        """Load vector store if needed, then return a retriever."""
        if self.embedding_manager.vector_store is None:
            # You need a way to know the path to load from
            vs_path = getattr(self.embedding_manager, 'persist_directory', None)
            if vs_path:
                self.embedding_manager.load_vector_store(vs_path)
            else:
                raise RuntimeError("Vector store not initialized and no path known")

        # Call the vector_store’s as_retriever with your kwargs
        retriever = self.embedding_manager.vector_store.as_retriever(**self.retriever_kwargs)
        return retriever

    def retrieve(self, question: str) -> pd.DataFrame:
        """
        Given a question, return the metadata + text of retrieved documents as a DataFrame.
        """
        retriever = self.get_retriever()
        # Use the retriever to fetch relevant documents; this returns a list of Document
        docs: List[Document] = retriever.get_relevant_documents(question)

        records = []
        for doc in docs:
            rec = dict(doc.metadata)  # metadata of the document
            # include the content/text (so you can show snippet)
            rec["text"] = doc.page_content
            records.append(rec)

        # convert list of dicts to DataFrame (if records is empty, returns empty DataFrame)
        return pd.DataFrame(records)
