import streamlit as st
import pandas as pd
from src.ingestion import parse_and_clean_csv, validate_csv_structure

# Page config
st.set_page_config(
    page_title="FinSense AI",
    page_icon="💰",
    layout="wide"
)

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
    # Read and display preview
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ File uploaded successfully!")
        
        # Validate CSV structure
        is_valid, error_msg = validate_csv_structure(df)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            st.info("Please ensure your CSV has the required columns: date, merchant, amount")
        else:
            # Show file info
            st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
            
            # Preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), width='stretch')
            
            # Confirm and process button
            if st.button("✓ Confirm and Process", type="primary"):
                with st.spinner("Processing and cleaning data..."):
                    # Parse and clean data
                    df_clean, summary = parse_and_clean_csv(df)
                    
                    # Store in session state
                    st.session_state['df_clean'] = df_clean
                    st.session_state['summary'] = summary
                    st.session_state['processed'] = True
                    
                st.success("✅ Data processed successfully!")
                st.rerun()
            
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")

# Show summary if data has been processed
if st.session_state.get('processed', False):
    st.divider()
    st.header("📊 Data Summary")
    
    summary = st.session_state['summary']
    df_clean = st.session_state['df_clean']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", summary['total_transactions'])
    with col2:
        st.metric("Valid Rows", summary['valid_rows'])
    with col3:
        st.metric("Invalid Rows", summary['invalid_rows'])
    with col4:
        st.metric("Suspicious", summary['suspicious_count'])
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric("Total Amount", f"${summary['total_amount']:,.2f}")
    with col6:
        st.metric("Average Amount", f"${summary['avg_amount']:,.2f}")
    
    # Date range
    st.write(f"**Date Range:** {summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}")
    
    # Show cleaned data
    with st.expander("View Cleaned Data"):
        st.dataframe(df_clean, width='stretch')
else:
    st.info("👆 Please upload a CSV file to begin")
    
    # Show template link
    with st.expander("Need a template?"):
        st.write("Your CSV should have these columns:")
        st.code("date,merchant,amount,category", language="text")
