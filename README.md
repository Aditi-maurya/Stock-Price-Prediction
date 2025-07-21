# 📈 Stock Price Prediction using LSTM

A deep learning-based time series forecasting model to predict future stock prices using historical closing price data. This project leverages the power of Long Short-Term Memory (LSTM) networks to capture temporal dependencies in stock data.Stock price prediction is a classic problem in financial analytics, where the goal is to predict future stock prices based on historical data and other financial indicators. Predicting stock prices using deep learning is a challenging task, as stock prices are influenced by a multitude of factors including market trends, macroeconomic conditions, political events, and investor sentiment. Nevertheless, deep learning techniques—particularly those involving recurrent neural networks (RNNs), Long Short-Term Memory (LSTM) networks, and other deep learning models—have shown promise in modeling this complex, non-linear data.This project typically focuses on predicting the closing price of a stock, which is the price at which the stock trades at the end of the trading day.

---

## 🚀 Features

- 📊 **Historical Data Analysis** – Uses past stock price trends to predict future prices.
- 🔧 **Data Preprocessing** – Includes Min-Max scaling and sliding window technique for sequence generation.
- 🧠 **LSTM-based Deep Learning Model** – Built using stacked LSTM layers for sequential learning.
- 📉 **Model Training & Evaluation** – Trained on historical data with visual evaluation of results.
- 🔮 **Future Stock Price Forecasting** – Predicts next-day prices from the last known values.
- 📈 **Interactive Visualization** – Plots predicted vs actual stock prices for easy comparison.

---

## 🧰 Tech Stack

| Component       | Library          |
|----------------|------------------|
| Programming     | Python           |
| Notebook        | Jupyter Notebook |
| Data Handling   | Pandas, NumPy    |
| Visualization   | Matplotlib       |
| Preprocessing   | Scikit-learn     |
| Deep Learning   | TensorFlow, Keras|

---

 ⚙️ How It Works

1.  **Load Data**
   - Historical stock data is imported from a CSV file using `pandas`.

2.  **Preprocessing**
   - Extracts only the ‘Close’ price column.
   - Applies **MinMaxScaler** to normalize data between 0 and 1.
   - Generates input sequences (time windows) of 100 time steps each for LSTM.

3.  **Model Architecture**
   - 3 stacked LSTM layers with `return_sequences=True`
   - Dropout layers to reduce overfitting
   - Final Dense layer to output the next predicted price

4. **Model Training**
   - Trained on 80% of the data, validated on the remaining 20%.

5.  **Evaluation & Forecasting**
   - Generates predictions for the validation set.
   - Compares predicted vs actual prices using Matplotlib.
   - Predicts future stock price using the last 100 days of data.

---
 📸 Sample Output

| Actual vs Predicted Prices |
|----------------------------|
| ![Prediction Graph](sample_output.png) |

---

## 📁 Folder Structure


