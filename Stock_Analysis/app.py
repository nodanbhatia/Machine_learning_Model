import streamlit as st
from api import get_stock_data
from analysis import add_analysis, get_statistics

st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title(" STOCK DASHBOARD")
st.subheader("Analyze a Stock in Real Time")

symbol = st.selectbox("Select Stock", ["IBM", "AAPL", "MSFT", "GOOGL"])

if "stock_df" not in st.session_state:
    st.session_state.stock_df = None

if st.button("Get Stock Data"):
    with st.spinner("Fetching data from Alpha Vantage..."):
        try:
            raw_data = get_stock_data(symbol)
            st.session_state.stock_df = add_analysis(raw_data)
        except Exception as e:
            st.error(f"API Error: {e}")
            st.session_state.stock_df = None

if st.session_state.stock_df is not None:
    df = st.session_state.stock_df

    st.subheader("Key Statistics")
    stats = get_statistics(df)

    cols = st.columns(len(stats))
    for col, (metric_name, metric_val) in zip(cols, stats.items()):
        col.metric(label=metric_name, value=metric_val)

    st.subheader("Price History & Indicators")
    st.line_chart(df[["close", "SMA_20", "SMA_50"]])

    st.subheader("Raw Data & Indicators")
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)
