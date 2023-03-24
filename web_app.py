# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

length = 100
start = '2000-01-01'
end = '2023-02-28'

st.title('Stock Price prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
yfin.pdr_override()

# df = pdr.get_data_yahoo(user_input, start=start, end=end)
df = yfin.download(user_input, start = start, end = f'{date.today()}', progress = False)

st.subheader('Data from 2000 - today')
st.write(df.describe())

if st.checkbox("Show The stock Prices"):
        st.dataframe(df, width = 1200)

# visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Times chart with MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'b', label = '100 days MA')
plt.plot(ma200, 'r', label = '200 days MA')
plt.plot(df.Close, 'g')
plt.legend()
st.pyplot(fig)

# ex = pd.DataFrame(df, columns=['Close'])
# st.line_chart(ex)


# DATA PREPROCESSING
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range = (0, 1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100: i])
        y_train.append(data_training_array[i, 0])
        
x_train, y_train = np.array(x_train), np.array(y_train)

# LOADING THE MODEL
model = load_model('stock_pred_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])
        
# PREDICTING THE PRICE
x_test , y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaleri = scaler.scale_

scaler_factor = 1/scaleri[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor



st.subheader('Prediction vs original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
st.pyplot(fig2)

forecastDays = st.number_input(label = 'Forecast Days...', step=1)
if forecastDays < 0:
        st.write('ERROR: Number must be positive')

def get_pred(forecastDays):
        forecast = []
        length = 100

        df_reshape = pd.DataFrame(df['Close'])
        df_scaled = scaler.transform(df_reshape)
        first_eval_batch = df_scaled[-length:]
        current_batch = first_eval_batch.reshape((1,length,1))

        for i in range(forecastDays):
                current_pred = model.predict(current_batch)[0]
                forecast.append(current_pred)
                current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
                
        return forecast

fore = []
fore = get_pred(forecastDays)
fore = pd.DataFrame(fore)
fore = fore * scaler_factor


def show_pred(fore):
        for i in range(len(fore)):
                st.write(f'{format(fore.iloc[i, 0],".2f")} --- [DAY: {i + 1}]')
                
show_pred(fore)