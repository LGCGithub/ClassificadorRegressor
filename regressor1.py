import pandas as pd
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.models import load_model

import os.path
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

#print(train_samples)
#print(train_labels)



dataset = pd.read_csv('tar2_sinais_vitais_treino_com_label.txt', encoding='utf-8')

dataset = dataset.to_numpy()

train_samples = dataset[:, 3:6] # qPA, pulso, resp
train_labels = dataset[:, 6] # gravidade

scaler = MinMaxScaler(feature_range=(0, 1)) # Normalização
train_samples = scaler.fit_transform(train_samples)

model = Sequential([
    Dense(units=512, input_shape=(3, ), activation="relu"),
    Dense(units=256, activation="relu"),
    Dense(units=1, activation="relu")
])

model.summary()

model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss=root_mean_squared_error)
model.fit(x=train_samples, y=train_labels, validation_split=0.3, batch_size=10, epochs=300, shuffle=True, verbose=2)


dataset_teste = pd.read_csv('tar2_sinais_vitais_teste_com_label.txt', encoding='utf-8')

dataset_teste = dataset_teste.to_numpy()

test_samples = dataset_teste[:, 3:6] # qPA, pulso, resp
test_labels = dataset_teste[:, 6] # gravidade

scaler = MinMaxScaler(feature_range=(0, 1)) # Normalização
test_samples = scaler.fit_transform(test_samples)

predictions = model.predict(x=test_samples, batch_size=1, verbose=2)

predictions = predictions.reshape(1, -1)

print("RMSE: ", np.sqrt(np.mean(np.square(predictions - test_labels))))