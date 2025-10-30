import streamlit as st
import pandas as pd

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
        
        # Show file info
        st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
        
        # Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), width='stretch')
        
        # Confirm button placeholder for next step
        if st.button("✓ Confirm and Process", type="primary"):
            st.info("Processing functionality coming next...")
            
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
else:
    st.info("👆 Please upload a CSV file to begin")
    
    # Show template link
    with st.expander("Need a template?"):
        st.write("Your CSV should have these columns:")
        st.code("date,merchant,amount,category", language="text")
        st.write("See `data/sample_template.csv` for an example")
