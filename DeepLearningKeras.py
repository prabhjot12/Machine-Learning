import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.normalization import BatchNormalization


#Input file
hockeydata = pd.read_csv('preprocessed_datasets.csv')
np.random.permutation(hockeydata)
X_train = hockeydata[(hockeydata.DraftYear.isin([2004, 2005, 2006]))]
X_train = X_train.drop(['id', 'PlayerName', 'sum_7yr_TOI', 'DraftYear', 'Country', 'Overall', 'GP_greater_than_0'], 1)
X_test = hockeydata[(hockeydata.DraftYear.isin([2007]))]
X_test = X_test.drop(['id', 'PlayerName', 'sum_7yr_TOI', 'DraftYear', 'Country', 'Overall', 'GP_greater_than_0'], 1)
Y_train = pd.DataFrame(X_train.sum_7yr_GP)
X_train = X_train.drop('sum_7yr_GP', 1)
Y_test = pd.DataFrame(X_test.sum_7yr_GP)
X_test = X_test.drop('sum_7yr_GP', 1)
country_group_values = X_train.country_group.unique()
positions = X_train.Position.unique()

#Creation of Dummy Variables
def dummyvariables(X_matrix, country_group_values, positions):
    for country in country_group_values:
        X_matrix.insert(0, country, 0)
        for i in X_matrix.index:
            if (X_matrix['country_group'][i] == country):
                X_matrix.set_value(i, country, 1)
            else:
                X_matrix.set_value(i, country, 0)
    for position in positions:
        X_matrix.insert(0, position, 0)
        for i in X_matrix.index:
            if (X_matrix['Position'][i] == position):
                X_matrix.set_value(i, position, 1)
            else:
                X_matrix.set_value(i, position, 0)


dummyvariables(X_train, country_group_values, positions)
dummyvariables(X_test, country_group_values, positions)
X_train = X_train.drop(['country_group', 'Position'], 1)
X_test = X_test.drop(['country_group', 'Position'], 1)

def normalization(X_matrix):
    columns = X_matrix.columns
    for col in columns:
        if (len(X_matrix[col].unique()) != 2):
            if (len(X_matrix[col].unique()) == 1):
                X_matrix = X_matrix.drop(col, 1)
            else:
                X_matrix[col] = (X_matrix[col] - X_matrix[col].mean()) / np.sqrt(X_matrix[col].var())
    return X_matrix
#All the steps tried have been commented out and only the ones that give best result are kept
X_train = normalization(X_train)
X_test = normalization(X_test)
np.random.seed(7)
#sgd = SGD(0.01)
nFeatures=X_train.shape[1]
model=Sequential()
model.add(Dense(units=32,input_shape=(nFeatures,), activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Dense(units=32,kernel_initializer='uniform',activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(Dense(units=16,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='linear'))
#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='Adadelta', loss='mse')
H=model.fit(X_train.values, Y_train.values, epochs=80,verbose=0)
mse_value= model.evaluate(X_test.values, Y_test.values)
y_pred=model.predict(X_test.values)
print('Mean Squared error on test set is:'+str(mse_value))
