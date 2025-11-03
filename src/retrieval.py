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
        # Use invoke to get docs
        docs = retriever.invoke(question)
        records = []
        for doc in docs:
            md = dict(doc.metadata or {})
            # Ensure metadata keys exist with default if missing
            md.setdefault("amount", None)
            md.setdefault("date", None)
            md.setdefault("merchant", None)
            rec = {
                "merchant": md["merchant"],
                "amount": md["amount"],
                "date": md["date"],
                "text": doc.page_content
            }
            records.append(rec)
        return pd.DataFrame(records)
