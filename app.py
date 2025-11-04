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

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container spacing */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Question bubble - User */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Answer bubble - AI */
    .ai-message {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1f2937;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #1f2937;
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Divider spacing */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# Session state defaults
defaults = {
    'processed': False,
    'df_clean': None,
    'summary': None,
    'vector_store_path': None,
    'embedding_manager': None,
    'chat_history': []
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

def reset_conversation():
    """Reset chat history"""
    st.session_state['chat_history'] = []
    st.rerun()

# Header
st.title("💰 FinSense AI")
st.caption("Your intelligent financial assistant powered by AI")

st.divider()

# Upload Section
if not st.session_state['processed']:
    st.header("📁 Get Started")
    st.markdown("Upload your bank statement CSV file to begin analyzing your transactions")
    
    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=["csv"],
        help="File must contain: date, merchant, and amount columns"
    )

    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')

            is_valid, error_msg = validate_csv_structure(df)
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                st.success(f"✅ File loaded: **{len(df)}** rows, **{len(df.columns)}** columns")
                
                with st.expander("👀 Preview Your Data"):
                    st.dataframe(df.head(20), width='stretch')
                
                if st.button("🚀 Process & Analyze", type="primary", use_container_width=False):
                    with st.spinner("Processing your transactions..."):
                        try:
                            df_clean, summary = parse_and_clean_csv(df)
                        except Exception as e:
                            st.error(f"Error processing CSV: {e}")
                            st.stop()

                        st.session_state['df_clean'] = df_clean
                        st.session_state['summary'] = summary
                        st.session_state['processed'] = True
                        st.session_state['chat_history'] = []

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

                        st.success("✅ Ready! Start asking questions below.")
                        st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    else:
        st.info("💡 Upload a CSV file to unlock powerful financial insights")
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📊 Smart Analysis")
            st.caption("Instantly understand your spending patterns and trends")
        with col2:
            st.markdown("#### 🔍 Natural Search")
            st.caption("Ask questions in plain English about your transactions")
        with col3:
            st.markdown("#### 💡 AI Insights")
            st.caption("Get intelligent answers powered by advanced AI")

# Main App - Only show if data is processed
if st.session_state['processed']:
    
    # Summary Section
    st.header("📊 Transaction Overview")
    df_clean = st.session_state['df_clean']
    summary = st.session_state['summary']
    currency = _choose_currency(df_clean, summary)

    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{summary.get('total_transactions', 0):,}")
    with col2:
        total_amount = float(summary.get('total_amount', 0.0))
        st.metric("Total Amount", _format_amount(total_amount, currency))
    with col3:
        avg_amount = float(summary.get('avg_amount', 0.0))
        st.metric("Average Transaction", _format_amount(avg_amount, currency))
    with col4:
        st.metric("Valid Rows", f"{summary.get('valid_rows', 0):,}")

    date_range = summary.get('date_range', {})
    st.caption(f"📅 **Period:** {date_range.get('start')} to {date_range.get('end')}")

    with st.expander("📋 View All Transactions"):
        st.dataframe(df_clean, width='stretch', height=400)

    st.divider()
    
    # Chat Interface
    col_left, col_right = st.columns([5, 1])
    with col_left:
        st.header("💬 Chat with Your Data")
    with col_right:
        if st.session_state['chat_history']:
            if st.button("🗑️ Clear", width='stretch'):
                reset_conversation()

    # Display chat history
    if st.session_state['chat_history']:
        for i, chat in enumerate(st.session_state['chat_history']):
            # User question
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # AI answer
            st.markdown(f"""
            <div class="ai-message">
                <strong>FinSense AI:</strong><br><br>{chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show relevant transactions if available
            if not chat['transactions'].empty:
                with st.expander(f"📊 Related Transactions ({len(chat['transactions'])} found)", expanded=False):
                    st.dataframe(
                        chat['transactions'],
                        width='stretch',
                        height=min(300, len(chat['transactions']) * 35 + 38)
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)
    
    else:
        # Suggestions when no chat history
        st.markdown("### 💡 Try asking me:")
        col1, col2 = st.columns(2)
        
        suggestions = [
            ("💰 What's my total spending?", "total_spend"),
            ("🏪 How much did I spend at Starbucks?", "starbucks"),
            ("📈 Show my top 5 expenses", "top_5"),
            ("🔍 Find transactions over $100", "over_100"),
        ]
        
        for idx, (text, key) in enumerate(suggestions):
            col = col1 if idx % 2 == 0 else col2
            with col:
                if st.button(text, key=f"sug_{key}", width='stretch'):
                    st.session_state['suggestion_clicked'] = text.split("?")[0].split(" ", 1)[1] + "?"
                    st.rerun()

    # Input form
    st.markdown("---")
    with st.form(key="question_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question",
                placeholder="e.g., How much did I spend on groceries last month?",
                label_visibility="collapsed"
            )
        with col2:
            submit_button = st.form_submit_button("Ask 🚀", width='stretch', type="primary")

    if submit_button and user_question.strip():
        with st.spinner("🤔 Thinking..."):
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

                st.session_state['chat_history'].append({
                    'question': user_question,
                    'answer': answer,
                    'transactions': df_retrieved
                })
                
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Handle suggestion clicks
    if 'suggestion_clicked' in st.session_state:
        user_question = st.session_state['suggestion_clicked']
        del st.session_state['suggestion_clicked']
        
        with st.spinner("🤔 Thinking..."):
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

                st.session_state['chat_history'].append({
                    'question': user_question,
                    'answer': answer,
                    'transactions': df_retrieved
                })
                
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")