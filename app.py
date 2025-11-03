import streamlit as st
import pandas as pd
import os
from src.ingestion import parse_and_clean_csv, validate_csv_structure
from src.embeddings import EmbeddingManager
from src.retrieval import RetrieverService
from src.llm_chain import build_qa_chain, answer_question
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Page config
st.set_page_config(
    page_title="FinSense AI",
    page_icon="💰",
    layout="wide"
)

# Session state defaults
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = None
if 'vector_store_path' not in st.session_state:
    st.session_state['vector_store_path'] = None
if 'embedding_manager' not in st.session_state:
    st.session_state['embedding_manager'] = None
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None
if 'retriever_service' not in st.session_state:
    st.session_state['retriever_service'] = None

def _choose_currency(df_clean: pd.DataFrame, summary: dict) -> str:
    """Choose currency from df_clean or fall back to USD ($)."""
    if df_clean is not None:
        cols = [c.lower() for c in df_clean.columns]
        if 'currency' in cols:
            cur_vals = df_clean.iloc[:, cols.index('currency')].dropna().astype(str).str.strip()
            if len(cur_vals) > 0:
                val = cur_vals.iloc[0]
                if val:
                    return val
    if summary and isinstance(summary, dict):
        cur = summary.get('currency') or summary.get('curr') or summary.get('ccy')
        if cur:
            return str(cur)
    return '$'

def _format_amount(amount: float, currency: str) -> str:
    """Format amount with chosen currency. Use symbol if present, otherwise prefix code."""
    if currency is None:
        currency = '$'
    cur = str(currency).strip()
    if cur == '':
        cur = '$'
    symbols = set('$₵€£¥¢₹')
    if any((ch in symbols) for ch in cur):
        return f"{cur}{amount:,.2f}"
    if len(cur) <= 3:
        return f"{cur.upper()} {amount:,.2f}"
    return f"{cur} {amount:,.2f}"

# Title and welcome
st.title("💰 FinSense AI")
st.subheader("Your Financial Chat Assistant")

st.markdown("""
    Welcome! Upload your bank statement CSV to get started.
    
    **What you can do:**
    - Upload your transaction CSV file
    - Ask natural language questions about your spending
    - Get insights backed by your actual data
""")

st.divider()

# File upload section
st.header("📁 Upload Your Bank Statement")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file with columns: date, merchant, amount, category"
)

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        st.success("✅ File uploaded successfully!")

        is_valid, error_msg = validate_csv_structure(df)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            st.info("Please ensure your CSV has the required columns: date, merchant, amount")
        else:
            st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
            st.subheader("Data Preview")
            st.dataframe(df.head(10), width='stretch')

            if st.button("✓ Confirm and Process", type="primary"):
                with st.spinner("Processing and cleaning data..."):
                    try:
                        df_clean, summary = parse_and_clean_csv(df)
                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")
                        st.stop()

                    st.session_state['df_clean'] = df_clean
                    st.session_state['summary'] = summary
                    st.session_state['processed'] = True

                    try:
                        with st.spinner("Creating embeddings..."):
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
                            st.success("✅ Embeddings created and stored successfully!")
                    except Exception as e:
                        st.error(f"Error creating embeddings: {str(e)}")

                st.success(f"✅ Indexing complete! All {len(df)} records processed.")

    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")

# After processing, show summary & chat UI
if st.session_state.get('processed', False):
    st.divider()
    st.header("📊 Data Summary")

    summary = st.session_state['summary'] or {}
    df_clean = st.session_state['df_clean']
    currency = _choose_currency(df_clean, summary)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", summary.get('total_transactions', 0))
    with col2:
        st.metric("Valid Rows", summary.get('valid_rows', 0))
    with col3:
        st.metric("Invalid Rows", summary.get('invalid_rows', 0))
    with col4:
        st.metric("Suspicious", summary.get('suspicious_count', 0))

    col5, col6 = st.columns(2)
    total_amount = float(summary.get('total_amount', 0.0) or 0.0)
    avg_amount = float(summary.get('avg_amount', 0.0) or 0.0)
    with col5:
        st.metric("Total Amount", _format_amount(total_amount, currency))
    with col6:
        st.metric("Average Amount", _format_amount(avg_amount, currency))

    date_start = summary.get('date_range', {}).get('start')
    date_end = summary.get('date_range', {}).get('end')
    if date_start is not None and date_end is not None:
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end)
        st.write(f"**Date Range:** {date_start.strftime('%Y-%m-%d')} to {date_end.strftime('%Y-%m-%d')}")
    else:
        st.write("**Date Range:** N/A")

    with st.expander("View Cleaned Data"):
        st.dataframe(df_clean, width='stretch')

    # Query UI
    st.divider()
    st.header("💬 Ask a question about your spending")

    user_question = st.text_input("Ask a question about your spending…", "")
    submit_button = st.button("Submit")

    if submit_button and user_question.strip() != "":
        with st.spinner("Processing your question…"):
            try:
                embedding_manager = st.session_state['embedding_manager']
                vector_store_path = st.session_state['vector_store_path']
                # Setup retriever service
                retriever_service = RetrieverService(embedding_manager, k=5)
                st.session_state['retriever_service'] = retriever_service
                retriever = retriever_service.get_retriever()
                # Build QA chain
                qa_chain = build_qa_chain(retriever)
                st.session_state['qa_chain'] = qa_chain
                # Get answer
                answer = answer_question(qa_chain, user_question)
                st.success("Answer ready!")
                st.write(answer)
                # Show relevant transactions
                df_retrieved = retriever_service.retrieve(user_question)
                if not df_retrieved.empty:
                    st.subheader("Relevant Transactions")
                    st.dataframe(df_retrieved)
            except Exception as e:
                st.error(f"Failed to answer question: {e}")
    else:
        if submit_button:
            st.warning("Please enter a question before submitting.")

else:
    st.info("👆 Please upload a CSV file to begin")
    with st.expander("Need a template?"):
        st.write("Your CSV should have these columns:")
        st.code("date,merchant,amount,category,currency", language="text")
