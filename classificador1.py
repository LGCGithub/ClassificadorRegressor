import pandas as pd
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.models import load_model

import os.path

dataset = pd.read_csv('tar2_sinais_vitais_treino_com_label.txt', encoding='utf-8')

dataset = dataset.to_numpy()

train_samples = dataset[:, 3:6] # qPA, pulso, resp
train_labels = dataset[:, 7] - 1 # classe

onehot_encoder = OneHotEncoder(sparse=False)
train_labels_onehot = onehot_encoder.fit_transform(train_labels.reshape(len(train_labels), 1))

train_labels_num = train_labels
train_labels = train_labels_onehot

scaler = MinMaxScaler(feature_range=(0, 1)) # Normalização
train_samples = scaler.fit_transform(train_samples)

#print(train_samples)
#print(train_labels)

if os.path.isfile("modelos\classificador1.h5") is False:
    model = Sequential([
        Dense(units=512, input_shape=(3, ), activation="relu"),
        Dense(units=256, activation="relu"),
        Dense(units=4, activation="softmax")
    ])

    model.summary()

    model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_samples, y=train_labels, validation_split=0.3, batch_size=10, epochs=300, shuffle=True, verbose=2)

    model.save("modelos\classificador1.h5")
else:
    model = load_model("modelos\classificador1.h5")

predictions = model.predict(x=train_samples, batch_size=10, verbose=0)
predictions = np.argmax(predictions, axis=-1)

print(predictions)
print(train_labels_num)


print(np.sum(np.abs(predictions - train_labels_num))) # Absolute error