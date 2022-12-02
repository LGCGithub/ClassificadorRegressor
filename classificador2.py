import pandas # for data manipulation
import numpy # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models

import os
import sys

#--------------------------------------------------
#                  Classificador
#--------------------------------------------------

path = 'tar2_sinais_vitais_treino_com_label.txt'

fileData = pandas.read_csv(path, encoding = 'utf-8')
print(fileData)

# input:
X  = fileData[['qPA', 'pulso', 'resp']]

# output:
Y = fileData['risco'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Fit the model 
criterion = 'entropy'
max_depth = 7

model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
clf = model.fit(X_train, Y_train)

# Predict class labels on training data
pred_labels_new = model.predict(X_train)
# Predict class labels on a test data
pred_labels_te = model.predict(X_test)

# Tree summary and model evaluation metrics
print('*************** Tree Summary ***************')
print('Classes: ', clf.classes_)
print('Tree Depth: ', clf.tree_.max_depth)
print('No. of leaves: ', clf.tree_.n_leaves)
print('No. of features: ', clf.n_features_in_)
print('--------------------------------------------------------')
print("")
        
print('*************** Evaluation on Training Data ***************')
score_new = model.score(X_train, Y_train)
print('Accuracy Score: ', score_new)
# Look at classification report to evaluate the model
print(classification_report(Y_train, pred_labels_new))
print('--------------------------------------------------------')

print('*************** Evaluation on Test Data ***************')
score_te = model.score(X_test, Y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(Y_test, pred_labels_te))
print('--------------------------------------------------------')
print("")


path2 = 'tar2_sinais_vitais_teste_com_label.txt'

fileData2 = pandas.read_csv(path, encoding = 'utf-8')

# input:
new_X  = fileData2[['qPA', 'pulso', 'resp']]

# output:
new_Y = fileData2['risco'].values

pred_labels_new = model.predict(new_X)

print('*************** Evaluation on New Data ***************')
score_new = model.score(new_X, new_Y)
print('Accuracy Score: ', score_new)
# Look at classification report to evaluate the model
print(classification_report(new_Y, pred_labels_new))
print('--------------------------------------------------------')