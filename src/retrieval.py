#retrieval.py

from typing import Any, Dict, List, Tuple
import pandas as pd
from langchain_core.documents import Document

class RetrieverService:
    def __init__(self, embedding_manager, df_clean: pd.DataFrame, summary: Dict[str, Any], k: int = 5):
        self.embedding_manager = embedding_manager
        self.df_clean = df_clean
        self.summary = summary
        self.k = k

    def _is_aggregation_query(self, question: str) -> bool:
        """Detect if query needs all data (totals, averages, counts)."""
        q_lower = question.lower()
        agg_keywords = [
            'total', 'sum', 'all', 'entire', 'overall', 'complete',
            'average', 'mean', 'avg', 'count', 'how many', 'number of'
        ]
        return any(keyword in q_lower for keyword in agg_keywords)

    def _extract_merchant_filter(self, question: str) -> str:
        """Extract merchant name from question if present."""
        q_lower = question.lower()
        for _, row in self.df_clean.iterrows():
            merchant = str(row.get('merchant', '')).lower()
            if merchant and merchant in q_lower:
                return merchant
        return None

    def _filter_display_data(self, question: str) -> pd.DataFrame:
        """Filter display data based on question context."""
        q_lower = question.lower()
        
        # Check for specific merchant
        merchant_filter = self._extract_merchant_filter(question)
        if merchant_filter:
            filtered = self.df_clean[
                self.df_clean['merchant'].str.lower().str.contains(merchant_filter, na=False)
            ][['date', 'merchant', 'amount']].copy()
            return filtered
        
        # If asking about "all" or "total" - show all
        if any(word in q_lower for word in ['all', 'total', 'entire', 'overall', 'everything', 'complete']):
            return self.df_clean[['date', 'merchant', 'amount']].copy()
        
        # Default: return empty, will be filled by similarity search
        return pd.DataFrame()

    def get_retriever(self):
        if self.embedding_manager.vector_store is None:
            path = getattr(self.embedding_manager, 'persist_directory', None)
            if path:
                self.embedding_manager.load_vector_store(path)
            else:
                raise RuntimeError("Vector store not initialized and no path known")
        return self.embedding_manager.vector_store.as_retriever(search_kwargs={'k': self.k})

    def retrieve(self, question: str) -> Tuple[List[Document], pd.DataFrame]:
        """
        Retrieve documents for LLM and determine display table.
        - LLM gets all transactions for aggregation queries
        - Display table shows only relevant transactions
        """
        is_aggregation = self._is_aggregation_query(question)
        
        if is_aggregation:
            # Pass ALL transactions to LLM for calculation
            safe_docs = []
            for _, row in self.df_clean.iterrows():
                merchant = str(row.get('merchant', ''))
                description = row.get('description', '')
                text = f"{merchant} {description}".strip() if description else merchant
                
                metadata = {
                    "date": str(row.get('date', '')),
                    "merchant": merchant,
                    "amount": float(row.get('amount', 0.0))
                }
                safe_docs.append(Document(page_content=text, metadata=metadata))
            
            # Add summary as first document
            summary_text = (
                f"SUMMARY: Total transactions: {self.summary.get('total_transactions', 0)}, "
                f"Total amount: {self.summary.get('total_amount', 0.0)}, "
                f"Average amount: {self.summary.get('avg_amount', 0.0)}"
            )
            summary_doc = Document(
                page_content=summary_text,
                metadata={
                    "date": "summary",
                    "merchant": "SUMMARY",
                    "amount": self.summary.get('total_amount', 0.0)
                }
            )
            safe_docs.insert(0, summary_doc)
            
            # Filter display based on question context
            df_display = self._filter_display_data(question)
            
        else:
            # Use similarity search for specific queries
            retriever = self.get_retriever()
            docs = retriever.invoke(question)
            
            safe_docs = []
            for doc in docs:
                md = doc.metadata or {}
                safe_md = {
                    "amount": md.get("amount", 0.0),
                    "date": md.get("date", ""),
                    "merchant": md.get("merchant", ""),
                }
                safe_doc = Document(page_content=doc.page_content, metadata=safe_md)
                safe_docs.append(safe_doc)
            
            # For non-aggregation, show only retrieved results
            df_display = pd.DataFrame([
                {
                    "date": d.metadata.get("date"),
                    "merchant": d.metadata.get("merchant"),
                    "amount": d.metadata.get("amount")
                }
                for d in safe_docs
            ])

        return safe_docs, df_display