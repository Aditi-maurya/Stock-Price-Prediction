import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Streamlit UI components
st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock Symbol (e.g., GOOG)", "GOOG")

# Fetch historical stock data
st.write("Fetching data...")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Validate the stock symbol and download data
try:
    stock_data = yf.download(stock, start=start, end=end)
    if stock_data.empty:
        st.error("No data available for the stock symbol provided.")
    else:
        st.subheader("Stock Data")
        st.write(stock_data)

        # Load pre-trained model
        try:
            model = load_model("Latest_stock_price_model(1).keras")
        except Exception as e:
            st.error(f"Error loading model: {e}")
        
        # Plot stock data
        splitting_len = int(len(stock_data) * 0.7)
        x_test = pd.DataFrame(stock_data.Close[splitting_len:])

        def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
            fig = plt.figure(figsize=figsize)
            plt.plot(values, 'Orange', label='Predicted Prices')
            plt.plot(full_data.Close, 'b', label='Actual Prices')
        
            plt.legend()
            if extra_data:
                plt.plot(extra_dataset)
            return fig

        # Example plot usage
        st.subheader("Stock Price Visualization")
        st.pyplot(plot_graph(figsize=(10, 6), values=x_test.values, full_data=stock_data))
except Exception as e:
    st.error(f"Error fetching stock data: {e}")

st.subheader('Original close price and MA for 250 days')
stock_data['MA_for_250_days'] = stock_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6),stock_data['MA_for_250_days'],stock_data,0))

st.subheader('Original close price and MA for 200 days')
stock_data['MA_for_200_days'] = stock_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), stock_data['MA_for_200_days'],stock_data,0))

st.subheader('Original close price and MA for 100 days')
stock_data['MA_for_100_days'] = stock_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), stock_data['MA_for_100_days'],stock_data,0))

st.subheader('Original close price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), stock_data['MA_for_100_days'],stock_data,1,stock_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data , y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions':inv_pre.reshape(-1)
 },
    index = stock_data.index[splitting_len+100:]
)
st.subheader('Original values vs Predicted values')
st.write(ploting_data)

st.subheader('Original Close price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([stock_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original test data", "Predicted test data"])
st.pyplot(fig)



