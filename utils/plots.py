import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

def plot_line_chart(df, forecast_df, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label="Historical Data", color='blue')
    ax.plot(forecast_df.index, forecast_df['Forecast'], label="Forecasted Data", color='red', linestyle='--')
    ax.set_title(f"{ticker} Stock Price: Historical and 30-Day Forecast", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_bar_chart(forecast_df):
    fig = go.Figure(
        data=[go.Bar(
            x=forecast_df.index.strftime('%Y-%m-%d'),
            y=forecast_df['Forecast'],
            marker_color='lightgreen'
        )],
        layout=dict(
            xaxis_title="Date",
            yaxis_title="Forecasted Price",
            height=450,
            template="plotly_white"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
