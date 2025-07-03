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

### 📊 Stock Price Forecasting (XGBoost)
- Fetch historical data using `yfinance`
- Feature engineering on stock prices (lags, rolling stats, etc.)
- Train XGBoost model
- Predict next **30 days** of stock prices
- Visualize historical + forecasted prices
- Display forecast table with formatted results

---

## 🧪 How it Works

- **Data Fetching**: Uses `yfinance` to get historical stock data.
- **Feature Engineering**: Creates features like moving averages, RSI, lags, rolling std, etc.
- **Model**: XGBoost regression is trained on engineered features.
- **Prediction**: Model is used to predict the next 30 days.
- **Visualization**: Uses `matplotlib`, `seaborn`, and `plotly` for detailed charts.

---

## To-Do (Future Enhancements)

- Add LSTM/GRU models for better forecasting
- Add technical indicators as toggle options
- Allow download of predictions as CSV
- Multi-stock comparison
- Deploy on Streamlit Cloud

---

# 🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.



