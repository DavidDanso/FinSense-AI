#retrieval.py

from typing import Any, Dict, List, Tuple
import pandas as pd
from langchain_core.documents import Document


class RetrieverService:
    def __init__(self, embedding_manager, df_clean: pd.DataFrame, summary: Dict[str, Any], k: int = 10):
        self.embedding_manager = embedding_manager
        self.df_clean = df_clean
        self.summary = summary
        self.k = k

    def _is_broad_query(self, question: str) -> bool:
        q_lower = question.lower()
        broad_keywords = [
            'story', 'tell me', 'narrative', 'summary', 'overview', 'describe',
            'pattern', 'trend', 'habit', 'behavior', 'insight', 'analyze',
            'total', 'all', 'entire', 'overall', 'complete', 'everything',
            'average', 'mean', 'count', 'how many'
        ]
        return any(keyword in q_lower for keyword in broad_keywords)

    def _extract_merchant_filter(self, question: str) -> str:
        q_lower = question.lower()
        for _, row in self.df_clean.iterrows():
            merchant = str(row.get('merchant', '')).lower()
            if merchant and len(merchant) > 2 and merchant in q_lower:
                return merchant
        return None

    def _should_show_display_table(self, question: str) -> bool:
        q_lower = question.lower()
        no_table_keywords = ['story', 'tell me', 'narrative', 'describe']
        if any(keyword in q_lower for keyword in no_table_keywords):
            return False
        return True

    def _get_extra_columns(self) -> List[str]:
        """Get list of extra columns to preserve beyond date/merchant/amount."""
        base_cols = {'date', 'merchant', 'amount', 'is_suspicious'}
        extra = [col for col in self.df_clean.columns if col not in base_cols]
        return extra

    def _filter_display_data(self, question: str) -> pd.DataFrame:
        q_lower = question.lower()
        
        # Get columns to display
        display_cols = ['date', 'merchant', 'amount']
        extra_cols = self._get_extra_columns()
        display_cols.extend([col for col in extra_cols if col in self.df_clean.columns])
        
        merchant_filter = self._extract_merchant_filter(question)
        if merchant_filter:
            filtered = self.df_clean[
                self.df_clean['merchant'].str.lower().str.contains(merchant_filter, na=False)
            ][display_cols].copy()
            return filtered
        
        if any(word in q_lower for word in ['all', 'total', 'entire', 'overall', 'everything']):
            return self.df_clean[display_cols].copy()
        
        return pd.DataFrame()

    def get_retriever(self):
        if self.embedding_manager.vector_store is None:
            path = getattr(self.embedding_manager, 'persist_directory', None)
            if path:
                self.embedding_manager.load_vector_store(path)
            else:
                raise RuntimeError("Vector store not initialized")
        return self.embedding_manager.vector_store.as_retriever(search_kwargs={'k': self.k})

    def retrieve(self, question: str) -> Tuple[List[Document], pd.DataFrame]:
        is_broad = self._is_broad_query(question)
        show_table = self._should_show_display_table(question)
        
        if is_broad:
            safe_docs = []
            for _, row in self.df_clean.iterrows():
                merchant = str(row.get('merchant', ''))
                amount = float(row.get('amount', 0.0))
                date = str(row.get('date', ''))
                
                # Build text with all available info
                text_parts = [f"{merchant} - ${amount} on {date}"]
                
                # Add reference if exists
                for ref_col in ['reference', 'transaction_reference', 'ref']:
                    if ref_col in row and row[ref_col]:
                        text_parts.insert(0, f"Ref: {row[ref_col]}")
                        break
                
                text = " ".join(text_parts)
                
                metadata = {
                    "date": date,
                    "merchant": merchant,
                    "amount": amount
                }
                
                # Add all extra fields to metadata
                for col in self._get_extra_columns():
                    if col in row and pd.notna(row[col]):
                        metadata[col] = str(row[col])
                
                safe_docs.append(Document(page_content=text, metadata=metadata))
            
            summary_text = (
                f"Dataset Summary: {self.summary.get('total_transactions', 0)} transactions, "
                f"Total: ${self.summary.get('total_amount', 0.0):.2f}, "
                f"Average: ${self.summary.get('avg_amount', 0.0):.2f}, "
                f"Date Range: {self.summary.get('date_range', {}).get('start')} to {self.summary.get('date_range', {}).get('end')}"
            )
            summary_doc = Document(
                page_content=summary_text,
                metadata={"date": "summary", "merchant": "SUMMARY", "amount": self.summary.get('total_amount', 0.0)}
            )
            safe_docs.insert(0, summary_doc)
            
            if show_table:
                df_display = self._filter_display_data(question)
            else:
                df_display = pd.DataFrame()
            
        else:
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
                
                # Preserve extra fields
                for key in md:
                    if key not in safe_md:
                        safe_md[key] = md[key]
                
                safe_doc = Document(page_content=doc.page_content, metadata=safe_md)
                safe_docs.append(safe_doc)
            
            if show_table:
                # Build display dataframe with extra columns
                display_data = []
                for d in safe_docs:
                    row_data = {
                        "date": d.metadata.get("date"),
                        "merchant": d.metadata.get("merchant"),
                        "amount": d.metadata.get("amount")
                    }
                    # Add extra columns
                    for key in d.metadata:
                        if key not in row_data:
                            row_data[key] = d.metadata[key]
                    display_data.append(row_data)
                
                df_display = pd.DataFrame(display_data)
            else:
                df_display = pd.DataFrame()

        return safe_docs, df_display