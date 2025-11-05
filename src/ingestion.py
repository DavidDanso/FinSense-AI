#ingestion.py

import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any
import re


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase stripped strings."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df


def _is_safe_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate CSV is safe and appropriate for financial data."""
    
    if len(df) == 0:
        return False, "CSV file is empty"
    if len(df) > 100000:
        return False, "CSV file too large (max 100,000 rows)"
    
    if len(df.columns) > 50:
        return False, "Too many columns (max 50)"
    if len(df.columns) < 2:
        return False, "Too few columns (minimum 2 required)"
    
    for col in df.columns:
        col_str = str(col).lower()
        dangerous_patterns = [
            'javascript:', '<script', 'onerror=', 'onclick=', 
            'onload=', 'eval(', 'exec(', '__import__', 'system('
        ]
        if any(danger in col_str for danger in dangerous_patterns):
            return False, f"Potentially dangerous column name: {col}"
    
    sample = df.head(min(10, len(df)))
    for col in sample.columns:
        for val in sample[col]:
            if pd.isna(val):
                continue
            val_str = str(val).lower()
            dangerous_content = [
                '<script', 'javascript:', 'onerror=', 'onclick=',
                'onload=', '<iframe', 'eval(', 'exec('
            ]
            if any(danger in val_str for danger in dangerous_content):
                return False, "Potentially malicious content detected"
    
    return True, ""


def _infer_date_column(df: pd.DataFrame) -> str:
    """Find date column from common aliases."""
    date_aliases = [
        'date', 'transaction_date', 'trans_date', 'posting_date', 
        'value_date', 'effective_date', 'statement_date', 'process_date',
        'date_posted', 'transaction_dt', 'trans_dt'
    ]
    
    cols = df.columns.tolist()
    
    for alias in date_aliases:
        if alias in cols:
            return alias
    
    for col in cols:
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(5)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() >= len(sample) * 0.8:
                        return col
            except:
                continue
    
    return None


def _infer_amount_column(df: pd.DataFrame) -> str:
    """Find amount column from common aliases."""
    amount_aliases = [
        'amount', 'transaction_amount', 'trans_amount', 'value',
        'debit', 'credit', 'debit_amount', 'credit_amount',
        'withdrawal', 'deposit', 'payment', 'charge',
        'withdrawals_and_other_subtractions', 'deposits_and_other_additions'
    ]
    
    cols = df.columns.tolist()
    
    for alias in amount_aliases:
        if alias in cols:
            return alias
    
    for col in cols:
        if col not in ['date', 'transaction_date']:
            try:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    numeric = pd.to_numeric(sample.astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce')
                    if numeric.notna().sum() >= len(sample) * 0.8:
                        return col
            except:
                continue
    
    return None


def _infer_merchant_column(df: pd.DataFrame) -> str:
    """Find merchant/description column from common aliases."""
    merchant_aliases = [
        'merchant', 'description', 'transaction_description', 'trans_description',
        'vendor', 'payee', 'store', 'shop', 'business', 'location',
        'merchant_name', 'store_name', 'transaction_details', 'details',
        'desc', 'transaction', 'name', 'counterparty', 'beneficiary',
        'account_name'
    ]
    
    cols = df.columns.tolist()
    
    for alias in merchant_aliases:
        if alias in cols:
            return alias
    
    for col in cols:
        if col not in ['date', 'amount'] and df[col].dtype == 'object':
            return col
    
    return None


def _merge_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Merge debit/credit columns into single amount column."""
    df = df.copy()
    
    has_debit = 'debit' in df.columns
    has_credit = 'credit' in df.columns
    
    if has_debit and has_credit:
        df['debit'] = _clean_amount_column(df['debit'])
        df['credit'] = _clean_amount_column(df['credit'])
        
        df['amount'] = df['credit'].fillna(0) - df['debit'].fillna(0)
        
        df = df.drop(columns=['debit', 'credit'])
    
    return df


def _clean_amount_column(series: pd.Series) -> pd.Series:
    """Clean and standardize amount values."""
    def clean_value(v: Any):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if s == "" or s.lower() in ['na', 'n/a', 'null', 'none']:
            return None
        
        if s.startswith('(') and s.endswith(')'):
            s = "-" + s[1:-1]
        
        for ch in ['$', '₵', 'gh₵', 'ghs', 'usd', 'ghc', 'eur', '€', '£', 'gbp', ',', ' ']:
            s = s.lower().replace(ch, '')
        
        try:
            return float(s)
        except:
            s2 = re.sub(r'[^0-9.\-]', '', s)
            try:
                return float(s2) if s2 not in ("", ".", "-") else None
            except:
                return None
    
    return series.apply(clean_value)


def parse_and_clean_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse and clean CSV with robust validation."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    is_safe, safety_msg = _is_safe_csv(df)
    if not is_safe:
        raise ValueError(f"Security validation failed: {safety_msg}")
    
    df_clean = _normalize_column_names(df)
    original_count = len(df_clean)
    
    df_clean = _merge_debit_credit(df_clean)
    
    date_col = _infer_date_column(df_clean)
    amount_col = _infer_amount_column(df_clean)
    
    if not date_col:
        raise ValueError("Could not identify date column. Please ensure your CSV has a date field.")
    if not amount_col:
        raise ValueError("Could not identify amount column. Please ensure your CSV has a transaction amount field.")
    
    if date_col != 'date':
        df_clean = df_clean.rename(columns={date_col: 'date'})
    if amount_col != 'amount':
        df_clean = df_clean.rename(columns={amount_col: 'amount'})
    
    merchant_col = _infer_merchant_column(df_clean)
    if merchant_col and merchant_col != 'merchant':
        df_clean = df_clean.rename(columns={merchant_col: 'merchant'})
    elif 'merchant' not in df_clean.columns:
        df_clean['merchant'] = 'unknown_merchant'
    
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    df_clean['amount'] = _clean_amount_column(df_clean['amount'])
    
    df_clean['merchant'] = (
        df_clean['merchant']
        .fillna('unknown_merchant')
        .astype(str)
        .str.strip()
        .str.lower()
        .replace('', 'unknown_merchant')
    )
    
    df_clean = df_clean.dropna(subset=['date', 'amount']).reset_index(drop=True)
    
    df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
    df_clean = df_clean.dropna(subset=['amount']).reset_index(drop=True)
    
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
        avg_amount = float(df_clean['amount'].mean())
        summary = {
            'total_rows': original_count,
            'valid_rows': len(df_clean),
            'invalid_rows': original_count - len(df_clean),
            'total_transactions': len(df_clean),
            'total_amount': total_amount,
            'avg_amount': avg_amount,
            'suspicious_count': int(df_clean['is_suspicious'].sum()),
            'date_range': {
                'start': str(df_clean['date'].min()),
                'end': str(df_clean['date'].max())
            }
        }
    
    return df_clean, summary


def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate CSV structure before processing."""
    if not isinstance(df, pd.DataFrame):
        return False, "Uploaded file could not be parsed as a CSV table."
    
    is_safe, safety_msg = _is_safe_csv(df)
    if not is_safe:
        return False, safety_msg
    
    df_normalized = _normalize_column_names(df)
    
    df_normalized = _merge_debit_credit(df_normalized)
    
    date_col = _infer_date_column(df_normalized)
    amount_col = _infer_amount_column(df_normalized)
    
    if not date_col:
        return False, "Could not identify a date column. Please ensure your CSV contains transaction dates."
    if not amount_col:
        return False, "Could not identify an amount column. Please ensure your CSV contains transaction amounts."
    
    return True, ""