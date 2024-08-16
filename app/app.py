import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yfinance as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start = '2015-01-01'
end = '2024-08-15'

st.title('Stock Trend Prediction')

user_inp = st.text_input('Enter Stock', 'AAPL')
df = data.download(user_inp, start=start, end=end)

st.subheader('Data from 2015 - 2024')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, 'r', label='100MA')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'b', label='Closing Price')
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.legend()
st.pyplot(fig)

# Splitting the data into training and testing
train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

st.write(f"Train data shape: {train.shape}")
st.write(f"Test data shape: {test.shape}")

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_arr = scaler.fit_transform(train)

# Load the pre-trained model
model = load_model('D:\\aryan\\stok\\app\\stock_model_50_time.h5')

# Prepare the testing data
past_100_days = train.tail(100)
final_df = pd.concat([past_100_days, test], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting the prices
y_predicted = model.predict(x_test)

# Scaling back to original values
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting predictions vs original
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
