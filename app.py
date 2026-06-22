"""Streamlit UI for the NASDAQ-10 multi-agent research application."""

import streamlit as st

from stock_agent.config import get_settings
from stock_agent.supervisor import build_supervisor
from stock_agent.tools.stock_data import YahooStockTool


st.set_page_config(page_title="NASDAQ-10 AI Research", page_icon="📈", layout="wide")
settings = get_settings()

st.title("NASDAQ-10 AI Stock Research")
st.caption(
    "Research prices and financial reports with specialist agents. "
    "For informational purposes only—not investment advice."
)

with st.sidebar:
    st.header("Research scope")
    selected = st.multiselect(
        "Companies",
        settings.tickers,
        default=list(settings.tickers[:3]),
    )
    st.markdown("**Available agents**")
    st.write("Supervisor · RAG · Stock Data")
    if not settings.reports_bucket:
        st.info("Set REPORTS_BUCKET to enable S3 financial-report retrieval.")

tab_research, tab_market = st.tabs(["Agent research", "NASDAQ-10 dashboard"])

with tab_research:
    uploaded_file = st.file_uploader(
        "Upload a PDF or text financial report",
        type=["pdf", "txt", "md"],
    )
    question = st.text_area(
        "Question",
        placeholder="Compare NVDA and MSFT prices and revenue risks.",
        height=110,
    )
    if st.button("Run research", type="primary", disabled=not question.strip()):
        with st.spinner("Supervisor is coordinating agents…"):
            result = build_supervisor().run(
                question=question,
                uploaded_content=uploaded_file.getvalue() if uploaded_file else None,
                uploaded_content_type=uploaded_file.type if uploaded_file else "text/plain",
                selected_tickers=tuple(selected) or settings.tickers,
            )
        st.markdown(result.answer)
        if result.sources:
            with st.expander("Sources"):
                for source in result.sources:
                    st.write(source)

with tab_market:
    if st.button("Refresh NASDAQ-10 prices"):
        with st.spinner("Loading Yahoo Finance data…"):
            frame = YahooStockTool(settings).universe_snapshot()
        if not frame.empty:
            st.dataframe(
                frame,
                column_config={
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "previous_close": st.column_config.NumberColumn(
                        "Previous close", format="$%.2f"
                    ),
                    "market_cap": st.column_config.NumberColumn(
                        "Market cap", format="$%.0f"
                    ),
                },
                width="stretch",
                hide_index=True,
            )
            if "price" in frame:
                chart_data = frame.dropna(subset=["price"]).set_index("ticker")[["price"]]
                if not chart_data.empty:
                    st.bar_chart(chart_data)
        else:
            st.info("Yahoo Finance did not return any stock data.")
