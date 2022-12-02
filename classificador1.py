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

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

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

from keras import backend as K

def f1_weighted(true, pred): #shapes (batch, 4)

    #for metrics include these two lines, for loss, don't include them
    #these are meant to round 'pred' to exactly zeros and ones
    #predLabels = K.argmax(pred, axis=-1)
    #pred = K.one_hot(predLabels, 4) 


    ground_positives = K.sum(true, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
        #all with shape (4,)
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
    weighted_f1 = K.sum(weighted_f1)

    
    return weighted_f1 #for metrics, return only 'weighted_f1'

model = Sequential([
    Dense(units=512, input_shape=(3, ), activation="relu"),
    Dense(units=256, activation="relu"),
    Dense(units=4, activation="softmax")
])

model.summary()

model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', f1_weighted])
model.fit(x=train_samples, y=train_labels, validation_split=0.3, batch_size=10, epochs=300, shuffle=True, verbose=2)

dataset_teste = pd.read_csv('tar2_sinais_vitais_teste_com_label.txt', encoding='utf-8')

dataset_teste = dataset_teste.to_numpy()

test_samples = dataset_teste[:, 3:6] # qPA, pulso, resp
test_labels = dataset_teste[:, 7] - 1 # classe

test_labels_onehot = onehot_encoder.fit_transform(test_labels.reshape(len(test_labels), 1))

test_labels_num = test_labels
test_labels = test_labels_onehot

scaler = MinMaxScaler(feature_range=(0, 1)) # Normalização
test_samples = scaler.fit_transform(test_samples)

predictions = model.predict(x=test_samples, batch_size=1, verbose=2)

from sklearn.metrics import accuracy_score

predictions = np.argmax(predictions, axis=1) 
test_labels_onehot = np.argmax(test_labels_onehot, axis=1)

print("Accuracy: ", accuracy_score(test_labels_onehot, predictions))
print("f-measure (weighted):", f1_score(test_labels_onehot, predictions, average="weighted"))