# embeddings.py

import os
import shutil
from typing import List, Dict, Any
import pandas as pd
import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import inspect
import math

load_dotenv()


class EmbeddingManager:
    def __init__(self, google_api_key: str):
        """Initialize with Google embeddings and placeholder for vector store."""
        if not google_api_key:
            raise ValueError("google_api_key must be provided to EmbeddingManager")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key,
            task_type="retrieval_document"
        )
        self.vector_store = None

    def _serialize_date(self, date_value: Any) -> Any:
        """Convert date or timestamp types to ISO string (or fallback)."""
        if date_value is None:
            return None
        if pd.isna(date_value):
            return None
        if isinstance(date_value, (pd.Timestamp, datetime.datetime, datetime.date)):
            return pd.to_datetime(date_value).isoformat()
        try:
            parsed = pd.to_datetime(str(date_value))
            return parsed.isoformat()
        except Exception:
            return str(date_value)

    def _serialize_amount(self, amount_value: Any) -> Any:
        """Convert numeric or string amount to float or int, or None."""
        if amount_value is None:
            return None
        try:
            val = float(amount_value)
            if val.is_integer():
                return int(val)
            return val
        except Exception:
            return None

    def _make_text_for_embedding(self, item: Dict[Any, Any]) -> str:
        """Compose a short text from merchant + description for embedding."""
        merchant = item.get("merchant", "") or ""
        description = item.get("description", "") or ""
        text = f"{merchant} {description}".strip()
        if text == "":
            text = "unknown"
        return text

    def _make_metadata(self, item: Dict[Any, Any]) -> Dict[str, Any]:
        """Prepare metadata dict for embedding record."""
        metadata = {
            "date": self._serialize_date(item.get("date", None)),
            "merchant": str(item.get("merchant", "")),
            "amount": self._serialize_amount(item.get("amount", None))
        }
        # you can add more metadata fields as needed
        return metadata

    def create_embeddings(
        self,
        data: List[Dict[Any, Any]],
        batch_size: int = 100,
        persist_path: str = None
    ) -> None:
        """
        Generate embeddings in batches for large data.
        Optionally persist vector store incrementally if persist_path given.
        """
        n = len(data)
        if n == 0:
            return

        num_batches = math.ceil(n / batch_size)
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n)
            batch = data[start:end]

            # Prepare texts & metadata lists
            texts = [self._make_text_for_embedding(item) for item in batch]
            metadatas = [self._make_metadata(item) for item in batch]

            # Add embeddings & metadata to vector_store
            if self.vector_store is None:
                # First batch: create vector_store from texts
                try:
                    self.vector_store = FAISS.from_texts(
                        texts=texts,
                        metadatas=metadatas,
                        embeddings=self.embeddings,
                    )
                except TypeError:
                    # fallback signature
                    self.vector_store = FAISS.from_texts(
                        texts, self.embeddings, metadatas=metadatas
                    )
            else:
                # For subsequent batches: add more docs
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )

            # Persist after each batch if path provided
            if persist_path:
                self.save_vector_store(persist_path)

            print(f"Batch {i+1}/{num_batches} done: items {start}–{end - 1}")

    def save_vector_store(self, path: str) -> None:
        """Save FAISS vector store to disk atomically."""
        if self.vector_store is None:
            raise RuntimeError("No vector store to save.")
        path = os.path.abspath(path)
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)

        tmp_dir = f"{path}.tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            self.vector_store.save_local(tmp_dir)
            if os.path.exists(path):
                shutil.rmtree(path)
            shutil.move(tmp_dir, path)
        except Exception as e:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            raise RuntimeError(f"Failed to save vector store: {e}") from e

    def load_vector_store(self, path: str) -> None:
        """Load existing FAISS vector store from disk, handling compatibility."""
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Vector store not found: {path}")
        if not any(os.scandir(path)):
            raise RuntimeError(f"Vector store directory empty or corrupt: {path}")

        try:
            sig = inspect.signature(FAISS.load_local)
            if "embeddings" in sig.parameters:
                self.vector_store = FAISS.load_local(path, embeddings=self.embeddings)
            elif "embedding" in sig.parameters:
                self.vector_store = FAISS.load_local(path, embedding=self.embeddings)
            else:
                self.vector_store = FAISS.load_local(path, self.embeddings)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vector store from '{path}': {e}"
            ) from e

    def get_retriever(self, **kwargs):
        """Return a retriever from the FAISS vector store."""
        if self.vector_store is None:
            raise RuntimeError("Vector store is not loaded/initialized.")
        return self.vector_store.as_retriever(**kwargs)
