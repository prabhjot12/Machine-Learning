import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Please insert the csv file in the below line
hockeydata = pd.read_csv('preprocessed_datasets.csv')
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

#Creation of interaction terms
def columnsMultiplied(X_matrix):
    for i in range(0, 22):
        for j in range(i + 1, 22):
            X_matrix.insert(22, X_matrix.columns[i] + X_matrix.columns[j], X_matrix.iloc[:, i] * X_matrix.iloc[:, j])
    return X_matrix


X_train = columnsMultiplied(X_train)
X_test = columnsMultiplied(X_test)

#Standardizing the columns
def normalization(X_matrix):
    columns = X_matrix.columns
    for col in columns:
        if (len(X_matrix[col].unique()) != 2):
            if (len(X_matrix[col].unique()) == 1):
                X_matrix = X_matrix.drop(col, 1)
            else:
                X_matrix[col] = (X_matrix[col] - X_matrix[col].mean()) / np.sqrt(X_matrix[col].var())
    return X_matrix


X_train = normalization(X_train)
X_test = normalization(X_test)
X_train.insert(0, 'Intercept', 1)
X_test.insert(0, 'Intercept', 1)
lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
validationErrorSet = list()

#Weight vector calculation function
def calc_weights(lambda_value, x_train, y_train):
    Xtranspose = np.matrix.transpose(x_train)
    Identity = np.identity(np.shape(x_train)[1])
    a = np.dot(Xtranspose, x_train)
    b = lambda_value * Identity
    add = np.add(a, b)
    inv = np.linalg.pinv(add)
    c = np.dot(Xtranspose, y_train)
    weight_vector = np.dot(inv, c)
    return weight_vector

#Sum squared error function
def calc_val_err(weight, x_test, y_test):
    ter = np.dot(x_test, weight)
    var = (y_test - ter) ** 2
    pred = var.sum()
    return (pred / 2)

#Creation of 10-folds and calculating validation errors
def CalculateErrorsKfold(lambda_val, X_train, Y_train):
    avg = 0
    N_train = int(len(X_train) / 10)
    for i in range(1, 11):
        if (i == 10):
            x_test = X_train.iloc[(i - 1) * N_train::]
            x_train = X_train[~X_train.isin(x_test)].dropna()
            y_test = Y_train.iloc[(i - 1) * N_train::]
            y_train = Y_train[~Y_train.isin(y_test)].dropna()
            weight_vector = calc_weights(lambda_val, x_train.values, y_train.values)
            error = calc_val_err(weight_vector, x_test.values, y_test.values)
            avg += error
        else:
            x_test = X_train.iloc[(i - 1) * N_train:i * N_train, :]
            x_train = X_train[~X_train.isin(x_test)].dropna()
            y_test = Y_train.iloc[(i - 1) * N_train:i * N_train, :]
            y_train = Y_train[~Y_train.isin(y_test)].dropna()
            weight_vector = calc_weights(lambda_val, x_train.values, y_train.values)
            error = calc_val_err(weight_vector, x_test.values, y_test.values)
            avg += error
    return avg / 10

#Calculating error on test set
def CalculateErrorsTestSet(lambda_val, X_train, Y_train, X_test, Y_test):
    weights = calc_weights(lambda_val, X_train.values, Y_train.values)

    error = calc_val_err(weights, X_test.values, Y_test.values)
    return error


temp = 0
for i in lambdas:
    error = CalculateErrorsKfold(i, X_train, Y_train)
    if ((temp != 0) & (error < temp)):
        best_error = error
        best_lambda = i
    temp = error
    validationErrorSet.append(error)

TestErrorSet = list()
temp_set = 0
for i in lambdas:
    error_test = CalculateErrorsTestSet(i, X_train, Y_train, X_test, Y_test)
    if ((temp_set != 0) & (error_test < temp_set)):
        best_error_data = error_test
        best_lambda_data = i
    temp_set = error_test
    TestErrorSet.append(error_test)

lmb = "Best Lambda With Cross Validation: " + str(best_lambda)
error = "Error at Best Lambda with Cross Validation: %.4f" % best_error

print(lmb)
print(error)
# Produce a plot of results.
# Change the details below as per your need.
plt.semilogx(lambdas, validationErrorSet, label='Validation error')
plt.semilogx(best_lambda, best_error, marker='o', color='r', label="Best Lambda Validation")
plt.semilogx(lambdas, TestErrorSet, label='Test set error')
plt.semilogx(best_lambda_data, best_error_data, marker='o', color='b', label="Best Lambda Test")
plt.ylabel('Sum Squared Error')
plt.legend()
plt.xlabel('Lambda')
plt.show()
