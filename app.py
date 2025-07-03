import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from utils.data_loader import load_stock_data
from utils.model import train_arima_and_forecast
from utils.plots import plot_line_chart, plot_bar_chart
from utils.downloader import get_forecast_csv

st.set_page_config(page_title="📊 Stock Dashboard", layout="wide")
tab1, tab2 = st.tabs(["📊 Stock Analysis", "📈 Stock Forecast"])

# ----------------------------- TAB 1: Stock Analysis -----------------------------
with tab1:
    st.title("📊 Stock Analysis Dashboard")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", key="analysis_ticker")
    with col2:
        start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"), key="analysis_start")
    with col3:
        end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"), key="analysis_end")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        st.subheader(f"About {info.get('shortName', ticker)}")
        st.markdown(f"**Description:** {info.get('longBusinessSummary', 'No description available.')}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
        st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '')})")

        summary_data = {
            "Market Cap": info.get("marketCap", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A")
        }
        st.subheader("📊 Market Summary")
        st.dataframe(pd.DataFrame(summary_data.items(), columns=["Metric", "Value"]))

        financials_data = {
            "Quick Ratio": info.get("quickRatio", "N/A"),
            "Revenue/Share": info.get("revenuePerShare", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
            "Debt to Equity": info.get("debtToEquity", "N/A"),
            "Return on Equity": info.get("returnOnEquity", "N/A")
        }
        st.subheader("📉 Financial Ratios")
        st.dataframe(pd.DataFrame(financials_data.items(), columns=["Ratio", "Value"]))

        data_daily = stock.history(period="2d")
        if len(data_daily) >= 2:
            change = data_daily['Close'].iloc[-1] - data_daily['Close'].iloc[-2]
            pct_change = (change / data_daily['Close'].iloc[-2]) * 100
            st.subheader("📌 Daily Change")
            st.metric(label="Price Change", value=f"{change:.2f}", delta=f"{pct_change:.2f}%")

        data_hist_table = stock.history(period="10d")
        st.subheader("📅 Last 10 Days Historical Data")
        st.dataframe(data_hist_table[['Open', 'High', 'Low', 'Close', 'Volume']].round(2))

        st.subheader("📈 Interactive Stock Chart")
        chart_col1, chart_col2, chart_col3 = st.columns([2, 2, 2])
        with chart_col1:
            chart_type = st.radio("Select Chart Type", ["Candlestick", "Line Chart"], key="chart_type")
        with chart_col2:
            time_range = st.selectbox("Select Time Range", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], key="time_range")
        with chart_col3:
            show_rsi = st.checkbox("Show RSI")
            show_macd = st.checkbox("Show MACD")

        data_hist = stock.history(period=time_range)
        if data_hist.empty:
            st.warning("No data for selected time range.")
        else:
            data_hist.index = data_hist.index.tz_localize(None)
            fig = go.Figure()

            if chart_type == "Line Chart":
                fig.add_trace(go.Scatter(x=data_hist.index, y=data_hist['Close'], mode='lines', name='Close'))
            else:
                fig.add_trace(go.Candlestick(
                    x=data_hist.index,
                    open=data_hist['Open'],
                    high=data_hist['High'],
                    low=data_hist['Low'],
                    close=data_hist['Close'],
                    name='Candlestick'))

            if show_rsi and len(data_hist) >= 14:
                delta = data_hist['Close'].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=data_hist.index, y=rsi, mode='lines', name='RSI', yaxis='y2'))

            if show_macd and len(data_hist) >= 26:
                exp1 = data_hist['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data_hist['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                fig.add_trace(go.Scatter(x=data_hist.index, y=macd, mode='lines', name='MACD', yaxis='y3'))
                fig.add_trace(go.Scatter(x=data_hist.index, y=signal, mode='lines', name='Signal Line', yaxis='y3'))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price",
                yaxis2=dict(title="RSI", overlaying='y', side='right', showgrid=False),
                yaxis3=dict(title="MACD", overlaying='y', side='right', position=0.95, showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")

# ----------------------------- TAB 2: ARIMA Forecast -----------------------------
with tab2:
    st.title("📈 Stock Price Forecasting ")

    ticker = st.text_input("Stock Ticker", value="AAPL", key="predict_ticker").upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"), key="predict_start")
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"), key="predict_end")
    st.markdown(
    """
    > ⚠️ **Disclaimer:** This forecast is for **academic and learning purposes only**. 
    It is **not intended for financial advice or real-world investment decisions.**
    """
)
    forecast_btn = st.button("📊 Forecast")

    if forecast_btn:
        try:
            df = load_stock_data(ticker, start_date, end_date)

            if df.empty:
                st.warning("⚠️ No data found. Please check the ticker or date range.")
            else:
                forecast_df = train_arima_and_forecast(df)

                st.subheader("📈 Line Chart: Historical + Forecasted")
                plot_line_chart(df, forecast_df, ticker)

                st.subheader("📊 Bar Chart: 30-Day Forecast")
                plot_bar_chart(forecast_df)

                st.subheader("📋 Forecast Table (Next 30 Business Days)")
                st.dataframe(forecast_df.reset_index().rename(columns={"index": "Date"}), use_container_width=True)

                csv = get_forecast_csv(forecast_df)
                st.download_button("⬇️ Download Forecast CSV", csv, f"{ticker}_forecast.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
