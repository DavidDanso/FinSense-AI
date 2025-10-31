# embeddings.py
import os
import shutil
from typing import List, Dict, Any
import pandas as pd
import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (safe at import)
load_dotenv()


class EmbeddingManager:
    def __init__(self, GOOGLE_API_KEY: str):
        """Initialize the embedding manager with Google's Generative AI model."""
        if not GOOGLE_API_KEY:
            raise ValueError("google_api_key must be provided to EmbeddingManager")

        # NOTE: Verify model name and arg names for your installed SDK version
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        self.vector_store = None

    def _serialize_date(self, date_value: Any) -> Any:
        """Convert various date types to ISO string or return None."""
        if date_value is None:
            return None
        # pandas NaT or NaN
        if pd.isna(date_value):
            return None
        # pd.Timestamp or datetime
        if isinstance(date_value, (pd.Timestamp, datetime.datetime, datetime.date)):
            return pd.to_datetime(date_value).isoformat()
        # string: try to parse
        try:
            parsed = pd.to_datetime(str(date_value))
            return parsed.isoformat()
        except Exception:
            return str(date_value)

    def _serialize_amount(self, amount_value: Any) -> Any:
        """Convert numeric types (numpy, pandas, strings) to plain Python float/int or None."""
        if amount_value is None:
            return None
        try:
            # handle pandas/numpy numeric types and strings that represent numbers
            val = float(amount_value)
            # if it is a whole number, keep as int for readability
            if val.is_integer():
                return int(val)
            return val
        except Exception:
            # fallback to None if it can't be parsed
            return None

    def create_embeddings(self, data: List[Dict[Any, Any]]) -> None:
        """Generate embeddings for transaction data and store them in FAISS."""
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for item in data:
            # Safely fetch fields
            merchant = item.get('merchant', '') or ''
            description = item.get('description', '') or ''
            date_raw = item.get('date', None)
            amount_raw = item.get('amount', None)

            # Combine merchant and description for embedding
            text = f"{str(merchant)} {str(description)}".strip() or "unknown"
            texts.append(text)

            # Serialize metadata to JSON-serializable primitives
            metadata = {
                'date': self._serialize_date(date_raw),
                'merchant': str(merchant),
                'amount': self._serialize_amount(amount_raw)
            }
            metadatas.append(metadata)

        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas
        )

        print(f"Indexing complete, {len(texts)} records processed")

    def save_vector_store(self, path: str) -> None:
        """Save the FAISS vector store to disk atomically."""
        if self.vector_store is None:
            raise RuntimeError("No vector store available to save.")

        path = os.path.abspath(path)
        # Ensure final directory exists (will be replaced)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        tmp_dir = f"{path}.tmp"
        # Clean tmp if exists
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            # Save into temporary directory first
            self.vector_store.save_local(tmp_dir)

            # Replace final directory atomically (remove if exists)
            if os.path.exists(path):
                shutil.rmtree(path)
            shutil.move(tmp_dir, path)
        except Exception as e:
            # Cleanup tmp and re-raise with context
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            raise RuntimeError(f"Failed to save vector store to '{path}': {e}") from e

    def load_vector_store(self, path: str) -> None:
        """Load the FAISS vector store from disk with validation and clear error messages."""
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Vector store directory not found at: {path}")

        # Basic sanity check: directory should not be empty
        if not any(os.scandir(path)):
            raise RuntimeError(f"Vector store directory is empty/corrupted: {path}")

        try:
            self.vector_store = FAISS.load_local(
                path,
                embeddings=self.embeddings
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vector store from '{path}'. Directory may be corrupted or incompatible. Original error: {e}"
            ) from e
