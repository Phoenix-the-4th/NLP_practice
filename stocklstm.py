# import pandas_datareader as pdr
# key = ""
# df = pdr.get_data_tiingo('AAPL', api_key = key)
# df.to_csv('AAPL.csv')

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


df = pd.read_csv('AAPL.csv')
df1 = df.reset_index()['close']
scaler = MinMaxScaler(feature_range = (0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

train_size = int(len(df1) * 0.7)
test_size = len(df1) - train_size
train_data, test_data = df1[:train_size, :], df1[train_size:, :1]

def create_dataset(dataset, time = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time - 1):
        a = dataset[i: (i + time), 0]
        dataX.append(a)
        dataY.append(dataset[i + time, 0])
    return np.array(dataX), np.array(dataY)

time = 100
X_train, y_train = create_dataset(train_data, time)
X_test, y_test = create_dataset(test_data, time)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (100, 1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()

model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=100, batch_size=64, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

print(math.sqrt(mean_squared_error(y_train, train_predict)))
print(math.sqrt(mean_squared_error(y_test, test_predict)))

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()