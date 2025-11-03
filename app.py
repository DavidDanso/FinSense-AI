#app.py

import streamlit as st
import pandas as pd
import os
from src.ingestion import parse_and_clean_csv, validate_csv_structure
from src.embeddings import EmbeddingManager
from src.retrieval import RetrieverService
from src.llm_chain import build_chain_only, answer_with_docs
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="FinSense AI", page_icon="💰", layout="wide")

# Session state defaults
defaults = {
    'processed': False,
    'df_clean': None,
    'summary': None,
    'vector_store_path': None,
    'embedding_manager': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _choose_currency(df_clean: pd.DataFrame, summary: dict) -> str:
    if df_clean is not None:
        cols = [c.lower() for c in df_clean.columns]
        if 'currency' in cols:
            cur_vals = df_clean.iloc[:, cols.index('currency')].dropna().astype(str).str.strip()
            if len(cur_vals) > 0:
                return cur_vals.iloc[0]
    if summary:
        for key in ('currency', 'curr', 'ccy'):
            if summary.get(key):
                return str(summary[key])
    return '$'

def _format_amount(amount: float, currency: str) -> str:
    if not currency:
        currency = '$'
    cur = str(currency).strip() or '$'
    symbols = set('$₵€£¥¢₹')
    if any(ch in cur for ch in symbols):
        return f"{cur}{amount:,.2f}"
    if len(cur) <= 3:
        return f"{cur.upper()} {amount:,.2f}"
    return f"{cur} {amount:,.2f}"

st.title("💰 FinSense AI")
st.subheader("Your Financial Chat Assistant")

st.markdown("""
Upload your transaction CSV, then ask questions about your spending, totals, trends, etc.
""")

st.divider()
st.header("📁 Upload Your Bank Statement (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Must have date, merchant, amount")

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.success("File uploaded successfully")

        is_valid, error_msg = validate_csv_structure(df)
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            st.dataframe(df.head(10), width='stretch')

            if st.button("✓ Confirm & Process", type="primary"):
                with st.spinner("Cleaning and indexing..."):
                    try:
                        df_clean, summary = parse_and_clean_csv(df)
                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")
                        st.stop()

                    st.session_state['df_clean'] = df_clean
                    st.session_state['summary'] = summary
                    st.session_state['processed'] = True

                    embedding_manager = EmbeddingManager(GOOGLE_API_KEY)
                    st.session_state['embedding_manager'] = embedding_manager

                    data = df_clean.to_dict('records')
                    base_dir = os.path.join(os.getcwd(), "data", "vectorstores")
                    os.makedirs(base_dir, exist_ok=True)
                    vector_store_path = os.path.join(base_dir, "default_index")

                    embedding_manager.create_embeddings(
                        data,
                        batch_size=200,
                        persist_path=vector_store_path
                    )
                    embedding_manager.save_vector_store(vector_store_path)
                    st.session_state['vector_store_path'] = vector_store_path

                    st.success("✅ Embeddings indexed")
    except Exception as e:
        st.error(f"Error reading upload: {e}")

if st.session_state['processed']:
    st.divider()
    st.header("📊 Data Summary")
    df_clean = st.session_state['df_clean']
    summary = st.session_state['summary']
    currency = _choose_currency(df_clean, summary)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Transactions", summary.get('total_transactions', 0))
    with c2:
        st.metric("Valid Rows", summary.get('valid_rows', 0))
    with c3:
        st.metric("Invalid Rows", summary.get('invalid_rows', 0))
    with c4:
        st.metric("Suspicious", summary.get('suspicious_count', 0))

    c5, c6 = st.columns(2)
    total_amount = float(summary.get('total_amount', 0.0))
    avg_amount = float(summary.get('avg_amount', 0.0))
    with c5:
        st.metric("Total Amount", _format_amount(total_amount, currency))
    with c6:
        st.metric("Average Amount", _format_amount(avg_amount, currency))

    date_range = summary.get('date_range', {})
    st.write(f"Date Range: {date_range.get('start')} → {date_range.get('end')}")

    with st.expander("View Cleaned Data"):
        st.dataframe(df_clean, width='stretch')

    st.divider()
    st.header("💬 Ask about your spending")

    user_question = st.text_input("Enter your question here")
    if st.button("Submit") and user_question.strip():
        with st.spinner("Generating answer..."):
            try:
                embedding_manager = st.session_state['embedding_manager']
                df_clean = st.session_state['df_clean']
                summary = st.session_state['summary']
                
                retriever_service = RetrieverService(
                    embedding_manager=embedding_manager,
                    df_clean=df_clean,
                    summary=summary
                )
                
                safe_docs, df_retrieved = retriever_service.retrieve(user_question)

                chain = build_chain_only()
                answer = answer_with_docs(chain, safe_docs, user_question)

                st.success("✅ Answer ready")
                st.write(answer)

                if not df_retrieved.empty:
                    st.subheader("Relevant Transactions")
                    st.dataframe(df_retrieved)
            except Exception as e:
                st.error(f"Failed to answer question: {e}")
    else:
        st.info("Enter a question and press Submit")
else:
    st.info("Upload your CSV to begin.")