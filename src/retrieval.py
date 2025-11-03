from typing import List, Dict, Any
import pandas as pd

class RetrieverService:
    def __init__(self, embedding_manager, **retriever_kwargs):
        """
        embedding_manager: instance of your EmbeddingManager with vector_store loaded or loadable.
        retriever_kwargs: passed to as_retriever (e.g., search_type, k)
        """
        self.embedding_manager = embedding_manager
        self.retriever_kwargs = retriever_kwargs

    def get_retriever(self):
        """Ensure vector store is loaded, then return retriever."""
        if self.embedding_manager.vector_store is None:
            path = getattr(self.embedding_manager, 'persist_directory', None)
            if path:
                self.embedding_manager.load_vector_store(path)
            else:
                raise RuntimeError("Vector store is not initialized or path unknown")

        return self.embedding_manager.vector_store.as_retriever(**self.retriever_kwargs)

    def retrieve(self, question: str) -> pd.DataFrame:
        """
        Retrieve top documents / embeddings for the question.
        Returns a DataFrame of metadata + text.
        """
        retriever = self.get_retriever()
        # ✅ Fixed: use invoke instead of get_relevant_documents
        docs = retriever.invoke(question)
        # each doc has .page_content, .metadata
        records = []
        for doc in docs:
            rec = dict(doc.metadata)
            rec["text"] = doc.page_content
            records.append(rec)
        return pd.DataFrame(records)
