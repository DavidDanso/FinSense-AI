import pandas as pd
from datetime import datetime
from typing import Tuple, Dict


def parse_and_clean_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse and clean transaction data.
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Tuple of (cleaned_df, summary_stats)
    """
    df_clean = df.copy()
    
    # Track original row count
    original_count = len(df_clean)
    
    # Parse date column
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Parse amount to numeric
    df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
    
    # Normalize merchant names
    if 'merchant' in df_clean.columns:
        df_clean['merchant'] = df_clean['merchant'].str.strip().str.lower()
    
    # Drop rows with invalid dates or amounts
    df_clean = df_clean.dropna(subset=['date', 'amount'])
    
    # Mark suspicious transactions (negative amounts)
    df_clean['is_suspicious'] = df_clean['amount'] < 0
    
    # Calculate summary statistics
    summary = {
        'total_rows': original_count,
        'valid_rows': len(df_clean),
        'invalid_rows': original_count - len(df_clean),
        'date_range': {
            'start': df_clean['date'].min(),
            'end': df_clean['date'].max()
        },
        'total_transactions': len(df_clean),
        'total_amount': df_clean['amount'].sum(),
        'avg_amount': df_clean['amount'].mean(),
        'suspicious_count': df_clean['is_suspicious'].sum()
    }
    
    return df_clean, summary


def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that CSV has required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['date', 'merchant', 'amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    return True, ""
