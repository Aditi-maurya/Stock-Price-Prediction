import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Check TensorFlow version
st.write(f"TensorFlow Version: {tf.__version__}")

# Streamlit UI components
st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID", "GOOG")

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
            model = load_model("C:\\Users\\Aditi Singh\\Downloads\\Latest_stock_price_model (1).keras")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Select the 'Close' column
        x_test = stock_data['Close'][int(len(stock_data) * 0.7):]

        # Plotting function
        def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, label="Close"):
            fig = plt.figure(figsize=figsize)
            plt.plot(values, 'orange', label='Predicted Prices')
            plt.plot(full_data['Close'], 'b', label='Actual Prices')
            plt.legend()
            if extra_data:
                plt.plot(extra_dataset, 'green', label='Extra Data')
                plt.legend()
            return fig

        # Example plot usage
        st.subheader("Stock Price Visualization")
        st.pyplot(plot_graph(figsize=(10, 6), values=x_test.values, full_data=stock_data))

        # Moving averages for 250, 200, and 100 days
        st.subheader('Original close price and MA for 250 days')
        stock_data['MA_for_250_days'] = stock_data['Close'].rolling(250).mean()
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_250_days'], stock_data))

        st.subheader('Original close price and MA for 200 days')
        stock_data['MA_for_200_days'] = stock_data['Close'].rolling(200).mean()
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_200_days'], stock_data))

        st.subheader('Original close price and MA for 100 days')
        stock_data['MA_for_100_days'] = stock_data['Close'].rolling(100).mean()
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_100_days'], stock_data))

        st.subheader('Original close price and MA for 100 days and MA for 250 days')
        st.pyplot(plot_graph((15, 6), stock_data['MA_for_100_days'], stock_data, 1, stock_data['MA_for_250_days']))

        # Scaling data for prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test.values.reshape(-1, 1))

        x_data = []
        y_data = []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Make predictions
        predictions = model.predict(x_data)

        # Inverse scaling for predictions and actual values
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        # Prepare data for plotting
        plotting_data = pd.DataFrame(
            {
                'Original Test Data': inv_y_test.flatten(),
                'Predictions': inv_pre.flatten()
            },
            index=stock_data.index[int(len(stock_data) * 0.7) + 100:]
        )

        st.subheader('Original values vs Predicted values')
        st.write(plotting_data)

        # Plotting original close price vs predicted close price
        st.subheader('Original Close price vs Predicted Close price')
        fig = plt.figure(figsize=(15, 6))
        plt.plot(stock_data['Close'], label="Original Data")
        plt.plot(plotting_data['Original Test Data'], label="Original Test Data")
        plt.plot(plotting_data['Predictions'], label="Predicted Test Data")
        plt.legend()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error fetching stock data: {e}")
