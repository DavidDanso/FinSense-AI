#ingestion.py

import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase stripped strings."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _infer_merchant_column(df: pd.DataFrame) -> str:
    """Try to find a merchant-like column if 'merchant' doesn't exist."""
    cols = df.columns.tolist()
    
    # Common merchant column aliases
    merchant_aliases = [
        'merchant', 'vendor', 'store', 'shop', 'business', 
        'description', 'desc', 'transaction', 'payee', 'name',
        'merchant_name', 'store_name', 'transaction_description'
    ]
    
    for alias in merchant_aliases:
        if alias in cols:
            return alias
    
    # If no match, use first string column after date/amount
    for col in cols:
        if col not in ['date', 'amount'] and df[col].dtype == 'object':
            return col
    
    return None


def _clean_amount_column(series: pd.Series) -> pd.Series:
    def clean_value(v: Any):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if s == "":
            return None
        if s.startswith('(') and s.endswith(')'):
            s = "-" + s[1:-1]
        for ch in ['$', '₵', 'gh₵', 'ghs', 'usd', 'ghc', ',', ' ']:
            s = s.replace(ch, '')
        try:
            return float(s)
        except Exception:
            import re
            s2 = re.sub(r'[^0-9\.\-]', '', s)
            try:
                return float(s2) if s2 not in ("", ".", "-") else None
            except Exception:
                return None

    return series.apply(clean_value)


def parse_and_clean_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df_clean = _normalize_column_names(df)

    # Check for date and amount first
    required_base = {'date', 'amount'}
    missing_base = required_base - set(df_clean.columns)
    if missing_base:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_base))}")

    # Try to find merchant column
    if 'merchant' not in df_clean.columns:
        merchant_col = _infer_merchant_column(df_clean)
        if merchant_col:
            df_clean = df_clean.rename(columns={merchant_col: 'merchant'})
        else:
            # Create a default merchant column with "Unknown"
            df_clean['merchant'] = 'unknown merchant'

    original_count = len(df_clean)

    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

    df_clean['amount'] = _clean_amount_column(df_clean['amount'])

    df_clean['merchant'] = df_clean['merchant'].fillna('unknown merchant').astype(str).str.strip().str.lower()
    df_clean['merchant'] = df_clean['merchant'].replace('', 'unknown merchant')

    df_clean = df_clean.dropna(subset=['date', 'amount']).reset_index(drop=True)

    df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')

    df_clean['is_suspicious'] = df_clean['amount'] < 0

    df_clean = df_clean.sort_values(by='date').reset_index(drop=True)

    if len(df_clean) == 0:
        summary = {
            'total_rows': original_count,
            'valid_rows': 0,
            'invalid_rows': original_count,
            'total_transactions': 0,
            'total_amount': 0.0,
            'avg_amount': 0.0,
            'suspicious_count': 0,
            'date_range': {'start': None, 'end': None}
        }
    else:
        total_amount = float(df_clean['amount'].sum())
        avg_amount = float(df_clean['amount'].mean()) if len(df_clean) > 0 else 0.0
        summary = {
            'total_rows': original_count,
            'valid_rows': len(df_clean),
            'invalid_rows': original_count - len(df_clean),
            'total_transactions': len(df_clean),
            'total_amount': total_amount,
            'avg_amount': avg_amount,
            'suspicious_count': int(df_clean['is_suspicious'].sum()),
            'date_range': {
                'start': pd.to_datetime(df_clean['date'].min()),
                'end': pd.to_datetime(df_clean['date'].max())
            }
        }

    return df_clean, summary


def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str]:
    if not isinstance(df, pd.DataFrame):
        return False, "Uploaded file could not be parsed as a CSV table."

    cols = [str(c).strip().lower() for c in df.columns]
    
    # Only require date and amount
    required_columns = ['date', 'amount']
    missing = [c for c in required_columns if c not in cols]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    return True, ""