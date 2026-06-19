"""Streamlit entry point for the Stock AI Agent."""

import streamlit as st

from stock_agent.supervisor import run_stock_research


st.set_page_config(page_title="Stock AI Agent", page_icon="📈", layout="wide")
st.title("Stock AI Research Agent")

uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
question = st.text_area("Ask about a stock, report, news, sentiment, or portfolio")

if st.button("Analyze", type="primary", disabled=not question.strip()):
    with st.spinner("Researching..."):
        result = run_stock_research(
            question=question,
            pdf_bytes=uploaded_file.getvalue() if uploaded_file else None,
        )
    st.markdown(result.answer)
    if result.sources:
        with st.expander("Sources"):
            for source in result.sources:
                st.write(f"- {source}")
