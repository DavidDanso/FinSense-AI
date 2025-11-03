from typing import Any, Dict, List
import pandas as pd
from langchain_core.documents import Document

class RetrieverService:
    def __init__(self, embedding_manager, **retriever_kwargs):
        self.embedding_manager = embedding_manager
        self.retriever_kwargs = retriever_kwargs

    def get_retriever(self):
        if self.embedding_manager.vector_store is None:
            path = getattr(self.embedding_manager, 'persist_directory', None)
            if path:
                self.embedding_manager.load_vector_store(path)
            else:
                raise RuntimeError("Vector store not initialized and no path known")
        return self.embedding_manager.vector_store.as_retriever(**self.retriever_kwargs)

    def retrieve(self, question: str) -> pd.DataFrame:
        retriever = self.get_retriever()
        docs = retriever.invoke(question)

        safe_docs = []
        for doc in docs:
            md = doc.metadata or {}
            # Ensure required metadata exists
            safe_md = {
                "amount": md.get("amount", 0.0),
                "date": md.get("date", ""),
                "merchant": md.get("merchant", ""),
            }
            # Patch document metadata to prevent missing-key errors
            safe_doc = Document(page_content=doc.page_content, metadata=safe_md)
            safe_docs.append(safe_doc)

        # Convert to dataframe for display
        df = pd.DataFrame([
            {
                "merchant": d.metadata.get("merchant"),
                "amount": d.metadata.get("amount"),
                "date": d.metadata.get("date"),
                "text": d.page_content
            }
            for d in safe_docs
        ])

        # Replace the internal docs list with safe_docs to fix missing metadata issue
        self.docs = safe_docs
        return df
