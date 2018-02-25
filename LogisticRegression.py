# Run logistic regression training for different learning rates with stochastic gradient descent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

# execfile("logistic_regression_mod.py")
hockeydata = pd.read_csv('preprocessed_datasets.csv')
X_train = hockeydata[(hockeydata.DraftYear.isin([2004, 2005, 2006]))]
np.random.seed(20)
np.random.permutation(X_train)
X_train = X_train.drop(['id', 'PlayerName', 'sum_7yr_TOI', 'DraftYear', 'Country', 'Overall', 'sum_7yr_GP'], 1)
X_test = hockeydata[(hockeydata.DraftYear.isin([2007]))]
X_test = X_test.drop(['id', 'PlayerName', 'sum_7yr_TOI', 'DraftYear', 'Country', 'Overall', 'sum_7yr_GP'], 1)
Y_train = pd.DataFrame(X_train.GP_greater_than_0)
X_train = X_train.drop('GP_greater_than_0', 1)
Y_test = pd.DataFrame(X_test.GP_greater_than_0)
X_test = X_test.drop('GP_greater_than_0', 1)
country_group_values = X_train.country_group.unique()
positions = X_train.Position.unique()


# Creation of Dummy Variables
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
            X_matrix[col] = (X_matrix[col] - X_matrix[col].mean()) / np.sqrt(X_matrix[col].var())
    return X_matrix


X_train = normalization(X_train)
X_test = normalization(X_test)
X_train.insert(0, 'Intercept', 1)
X_test.insert(0, 'Intercept', 1)
# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 300
tol = 0.0001

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
# etas=[0.001,0.005,0.0005,0.0001]


data = X_train.as_matrix()


# Target values, 0 for class 1, 1 for class 2.
def convert_output_boolean(row):
    return 1 if row.values[0] == 'yes' else 0


Y_train = Y_train.apply(convert_output_boolean, axis=1)
Y_test = Y_test.apply(convert_output_boolean, axis=1)

t = Y_train.as_matrix()
t_test = Y_test.as_matrix()
n_train = t.size
# Error values over all iterations.
all_errors = dict()

for eta in etas:
    # Initialize w.
    w = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    e_all = []

    for iter in range(0, max_iter):
        for n in range(0, n_train):
            # Compute output using current w on sample x_n.
            y = sps.expit(np.dot(data[n, :], w))

            # Gradient of the error, using Assignment result
            grad_e = (y - t[n]) * data[n, :]
            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w = w - ((eta * grad_e))

        # Compute error over all examples, add this error to the end of error vector.
        # Compute output using current w on all data X.
        y = sps.expit(np.dot(data, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = 0
        for index, j in enumerate(t):
            if j == 0:
                e += np.log(1 - y[index])
            else:
                e += np.log(y[index])
        e = -e / n_train
        e_all.append(e)
        # e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))
        # e_all.append(e)

        # Print some information.
        # print ('eta={0}, epoch {1:d},negative log-likelihood {2:.4f}, w={3}'.format(eta, iter, e, w.T))

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break
    accuracy = list()
    for row in range(0, len(X_test)):
        y_pred = sps.expit(np.dot(X_test.values[row], w))
        if (y_pred > 0.5):
            y_pred = 1
        else:
            y_pred = 0
        if (y_pred == t_test[row]):
            accuracy.append(1)
    acc = len(accuracy) / len(X_test)
    print("Accuracy for eta " + str(eta) + " is " + str(acc))
    all_errors[eta] = e_all

# Plot error over iterations for all etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
for eta in sorted(all_errors):
    plt.plot(all_errors[eta], label='sgd eta={}'.format(eta))

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression with SGD')
plt.xlabel('Epoch')
plt.legend()
plt.show()
