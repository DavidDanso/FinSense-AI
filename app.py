# app.py
import os
import re
import html
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# your project imports (keep these as before)
from src.ingestion import parse_and_clean_csv, validate_csv_structure
from src.embeddings import EmbeddingManager
from src.retrieval import RetrieverService
from src.llm_chain import build_chain_only, answer_with_docs

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="FinSense AI", page_icon="💰", layout="wide")

# ---- Improved CSS ----
st.markdown("""
<style>
    :root{
        --bg-card: #0f1724;
        --muted: #6b7280;
        --accent: #667eea;
        --accent-2: #7c3aed;
        --ai-bg: #ffffff;
        --ai-border: rgba(102,126,234,0.15);
        --radius-lg: 14px;
        --radius-sm: 8px;
    }

    .main > div {
        padding-top: 2rem;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: #0f1724;
    }

    .user-message{
        display:block;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 999px;
        margin: 0.75rem 0;
        box-shadow: 0 6px 18px rgba(102,126,234,0.12);
        max-width: 90%;
        word-break: break-word;
    }
    .user-message strong{ font-weight:700; margin-right:0.6rem; }

    .ai-message {
        background: var(--ai-bg);
        border: 1px solid var(--ai-border);
        border-radius: var(--radius-lg);
        padding: 1rem;
        margin: 0.9rem 0;
        color: #0b1220;
        box-shadow: 0 6px 24px rgba(12,18,28,0.06);
        max-width: 100%;
    }

    .ai-header {
        display:flex;
        gap:0.75rem;
        align-items:center;
        margin-bottom: 0.5rem;
    }
    .ai-avatar{
        width:36px;
        height:36px;
        border-radius:10px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
        display:inline-flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-weight:700;
        box-shadow: 0 6px 18px rgba(124,58,237,0.12);
    }
    .ai-title{
        font-weight:700;
        font-size:1rem;
        color:#0b1220;
    }
    .ai-sub{
        font-size:0.85rem;
        color: var(--muted);
        margin-left:6px;
    }

    .ai-body{ 
        line-height:1.6;
        font-size:0.98rem;
        color:#0b1220;
    }

    .amount-badge {
        display:inline-block;
        padding:0.18rem 0.5rem;
        border-radius:999px;
        background:rgba(102,126,234,0.1);
        color:var(--accent);
        font-weight:600;
        font-size:0.92rem;
        margin:0 0.15rem;
    }

    .ai-body ul, .ai-body ol {
        padding-left:1.25rem;
        margin:0.6rem 0;
    }
    .ai-body li { margin:0.35rem 0; }
    .ai-body blockquote {
        margin: 0.6rem 0;
        padding: 0.6rem 1rem;
        background: #f8fafc;
        border-left: 3px solid var(--accent);
        color: #111827;
        border-radius:8px;
    }
    .ai-body pre, .ai-body code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;
        background: #0f1724;
        color: #e6eef8;
        padding: 0.6rem;
        border-radius: 8px;
        overflow:auto;
        font-size:0.85rem;
    }

    .ai-body h2 { font-size:1.05rem; margin-top:0.6rem; }
    .ai-body h3 { font-size:0.98rem; margin-top:0.5rem; color:#0b1220; }

    .ai-meta { font-size:0.82rem; color:var(--muted); margin-top:0.6rem; }

    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton button { border-radius:8px; }

    .spacer { height: 12px; }
</style>
""", unsafe_allow_html=True)

# ---- Session state defaults ----
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
                    st.dataframe(df.head(20), use_container_width=True)
                
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
    
    st.header("📊 Transaction Overview")
    df_clean = st.session_state['df_clean']
    summary = st.session_state['summary']
    currency = _choose_currency(df_clean, summary)

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
        st.dataframe(df_clean, use_container_width=True, height=400)

    st.divider()
    
    col_left, col_right = st.columns([5, 1])
    with col_left:
        st.header("💬 Chat with Your Data")
    with col_right:
        if st.session_state['chat_history']:
            if st.button("🗑️ Clear", use_container_width=True):
                reset_conversation()

    # ---- Chat history rendering (improved) ----
    try:
        import markdown as md  # optional: pip install markdown
    except Exception:
        md = None

    def render_answer_as_html(raw_markdown: str) -> str:
        if not raw_markdown:
            return ""
        if md:
            html_body = md.markdown(raw_markdown, extensions=['extra', 'nl2br', 'sane_lists', 'fenced_code'])
        else:
            html_body = "<p>" + html.escape(raw_markdown).replace("\n", "<br>") + "</p>"

        # highlight currency-like amounts
        def amt_repl(m):
            return f'<span class="amount-badge">{m.group(0)}</span>'

        # regex: look for currency symbols followed by amounts
        html_body = re.sub(r'(?<!\w)(?:\$|£|€|₹|₵)\s?\d{1,3}(?:[,\d{3}]*)?(?:\.\d{1,2})?', amt_repl, html_body)
        return html_body

    if st.session_state['chat_history']:
        for i, chat in enumerate(st.session_state['chat_history']):
            # User bubble
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {html.escape(chat['question'])}
            </div>
            """, unsafe_allow_html=True)

            # AI card with converted body
            html_answer = render_answer_as_html(chat.get('answer', ''))
            st.markdown(f"""
            <div class="ai-message">
                <div class="ai-header">
                    <div class="ai-avatar">FA</div>
                    <div>
                        <div class="ai-title">FinSense AI <span class="ai-sub">• smart summary</span></div>
                        <div class="ai-meta">Answer generated from your uploaded transactions</div>
                    </div>
                </div>
                <div class="ai-body">
                    {html_answer}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not chat['transactions'].empty:
                with st.expander(f"📊 Related Transactions ({len(chat['transactions'])} found)", expanded=False):
                    st.dataframe(
                        chat['transactions'],
                        use_container_width=True,
                        height=min(300, len(chat['transactions']) * 35 + 38)
                    )

            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

    else:
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
                if st.button(text, key=f"sug_{key}", use_container_width=True):
                    st.session_state['suggestion_clicked'] = text.split("?")[0].split(" ", 1)[1] + "?"
                    st.rerun()

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
            submit_button = st.form_submit_button("Ask 🚀", use_container_width=True, type="primary")

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
