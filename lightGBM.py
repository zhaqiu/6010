import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
folder = 'C:/Users/user/PycharmProjects/HKUST5140/data/'


train = pd.read_csv(folder+'train_data2.csv')

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
y_pred=model.predict(x_test)
#convert into binary values
for i in range(0,len(y_pred)):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:
       y_pred[i]=0



#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
accuracy=accuracy_score(y_pred,y_test)
print(cm)
print(accuracy)



#
# Create the LightGBM data containers
#
# categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]
# train_data = lightgbm.Dataset(x, label=y, categorical_feature=categorical_features)
# test_data = lightgbm.Dataset(x_test, label=y_test)


# submission = pd.read_csv('C:/Users/user/PycharmProjects/HKUST5140/data/new_test.csv')
# ids = submission['id'].values
# submission.drop('id', inplace=True, axis=1)
#
#
# x = submission.values
# y = model.predict(x,raw_score=True)
#
# output = pd.DataFrame({'id': ids, 'target': y})
# output.to_csv('C:/Users/user/PycharmProjects/HKUST5140/data/submission.csv', index=False)




