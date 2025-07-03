# 📈 Stock Analysis and Forecasting App 

A powerful web-based application built with **Streamlit**, combining **real-time stock analysis** with **30-day stock price forecasting** using **XGBoost** and advanced feature engineering.

> ⚠️ **Disclaimer**: This application is built for **academic and educational purposes only**. The stock price predictions are not suitable for real-world trading or investment decisions. Use it solely for learning and experimentation.

---

## 🚀 Features

### 🔍 Stock Analysis Dashboard
- Company summary: name, sector, website, employee count, etc.
- Key market metrics: Market Cap, EPS, P/E Ratio, Beta
- Financial ratios: Quick Ratio, ROE, Debt/Equity, etc.
- Daily price change tracking
- Last 10 days historical data table
- Interactive charts:
  - Line or Candlestick
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)

### 📊 Stock Price Forecasting (ARIMA)
- Input stock ticker and date range
- Automatically fetch historical data using `yfinance`
- Fit ARIMA model to closing price data
- Forecast next **30 business days**
- Dual visualization:
  - 📈 Matplotlib Line Chart (historical + forecast)
  - 📊 Bar Chart of forecasted prices
- Forecast table with date-wise predicted values
- 📥 Download forecast as CSV

---

## 🧪 How it Works

- **Data Fetching**: Uses `yfinance` to download historical stock data based on user input.
- **Modeling**: Trains a basic ARIMA model using `statsmodels` to capture time-dependent trends.
- **Forecasting**: Predicts the next 30 business days and visualizes it.
- **Visualization**: Uses `matplotlib`, `seaborn`, and `plotly` to show insights clearly and interactively.

---

## To-Do (Future Enhancements)

- Add LSTM/GRU models for better forecasting
- Add technical indicators as toggle options
- Allow download of predictions as CSV
- Multi-stock comparison
- Deploy on Streamlit Cloud

---

# 🤝 Contributions
Contributions, suggestions, and pull requests are welcome!

- If you find a bug or want a feature, please open an issue.
- Want to add something cool? Fork and submit a pull request.



