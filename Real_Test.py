from sklearn.preprocessing import StandardScaler
import lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

folder = 'C:/Users/user/PycharmProjects/HKUST5140/data/'

# read the raw data
train = pd.read_csv(folder+'train_data.csv')
#KNN
y_train = train.target.values
train.drop(['target'], inplace=True, axis=1)

x_train = train.values

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

# read the test data
test = pd.read_csv(folder+'test.csv')

ids = test['id'].values
test.drop('id', inplace=True, axis=1)
x_test = test.values
x_train = scaler.transform(x_test)
y_pred = classifier.predict(x_test)

output = pd.DataFrame({'id': ids, 'target': y_pred})
output.to_csv(folder+'submission.csv', index=False)

#Light GBM

y = train.target.values
train.drop(['target'], inplace=True, axis=1)
x = train.values

#
# Create training and validation sets
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

d_train = lightgbm.Dataset(x_train, label=y_train)
d_test  = lightgbm.Dataset(x_test, label=y_test)
#
# Train the model
#

parameters = {
    'application': 'binary',
    'objective': 'binary', # binary classification (or logistic regression)
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31, #This is the main parameter to control the complexity of the tree model.
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20, #frequency for bagging， non-zero to enable bagging
    'learning_rate': 0.05, #shrinkage rate
    'verbose': 0
}
model = lightgbm.train(parameters,
                       d_train,
                       valid_sets=d_test,
                       num_boost_round=5000,
                       early_stopping_rounds=100)

#Prediction
submission = pd.read_csv(folder+'test.csv')
ids = submission['id'].values
submission.drop('id', inplace=True, axis=1)
x_real_test = submission.values
y_pred=model.predict(x_real_test)
#convert into binary values
for i in range(0,len(y_pred)):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:
       y_pred[i]=0

output = pd.DataFrame({'id': ids, 'target': y_pred})
output.to_csv('C:/Users/user/PycharmProjects/HKUST5140/data/submission_lightGBM.csv', index=False)

