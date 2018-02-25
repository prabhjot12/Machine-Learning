# Answer 7
import pandas as pd
import numpy as np
import math
# input the .csv file in the below line
hockeydata=pd.read_csv('/Users/gurbakshseehra/Desktop/preprocessed_datasets.csv')
x=0
for i in (hockeydata.Weight):
    x+=i
mean=x/hockeydata.Weight.count()
y=0
for j in (hockeydata.Weight):
    y+=(j-mean)**2
variance = y/hockeydata.Weight.count()
a=0
for k in hockeydata.Weight[hockeydata['GP_greater_than_0']=='yes']:
    a+=k
mean_weight_GP_grt_0_true = a/hockeydata.Weight[hockeydata['GP_greater_than_0']=='yes'].count()
b=0
for l in hockeydata.Weight[hockeydata['GP_greater_than_0']=='no']:
    b+=l
mean_weight_GP_grt_0_false = b/hockeydata.Weight[hockeydata['GP_greater_than_0']=='no'].count()
p=0
for t in hockeydata.Weight[hockeydata['GP_greater_than_0']=='yes']:
    p+=(t-mean_weight_GP_grt_0_true)**2
variance_weight_GP_grt_0_true= p/hockeydata.Weight[hockeydata['GP_greater_than_0']=='yes'].count()
q=0
for u in hockeydata.Weight[hockeydata['GP_greater_than_0']=='no']:
    q+=(u-mean_weight_GP_grt_0_false)**2
variance_weight_GP_grt_0_false= q/hockeydata.Weight[hockeydata['GP_greater_than_0']=='no'].count()
print('Mean value of weight : ' +str(mean))
print('Variance value of weight : ' +str(variance))
print('Mean with GP>0=true : ' +str(mean_weight_GP_grt_0_true))
print('Mean with GP>0=false : ' +str(mean_weight_GP_grt_0_false))
print('Variance with GP>0=true : ' +str(variance_weight_GP_grt_0_true))
print('Variance with GP>0=false : ' +str(variance_weight_GP_grt_0_false))

#----------------------------------------------------------------------------------------------------------------------------

# Answer 8
import pandas as pd
import numpy as np
import math
# input the .csv file in the below line
hockeydata = pd.read_csv('/Users/gurbakshseehra/Desktop/preprocessed_datasets.csv')
hockeydata = hockeydata.drop(['sum_7yr_GP', 'sum_7yr_TOI', 'Country'], 1)
X_train = pd.DataFrame(hockeydata[(hockeydata.DraftYear == 1998) | (hockeydata.DraftYear == 1999) | (hockeydata.DraftYear == 2000)].iloc[:,list(range(2, 22))])
X_train = X_train.drop('DraftYear', 1)
y_train = hockeydata[(hockeydata.DraftYear == 1998) | (hockeydata.DraftYear == 1999) | (hockeydata.DraftYear == 2000)].iloc[:,21].values
X_test = hockeydata[(hockeydata.DraftYear == 2001)].iloc[:, list(range(2, 21))]
X_test = X_test.drop('DraftYear', 1)
y_test = hockeydata[(hockeydata.DraftYear == 2001)].iloc[:, 21].values
num_GPgrt_0 = X_train['GP_greater_than_0'][X_train.GP_greater_than_0 == 'yes'].count()
num_GP_not_grt_0 = X_train['GP_greater_than_0'][X_train.GP_greater_than_0 == 'no'].count()
P_num_grt_0 = num_GPgrt_0 / X_train['GP_greater_than_0'].count()
P_num_not_grt_0 = num_GP_not_grt_0 / X_train['GP_greater_than_0'].count()
data_mean = X_train.groupby('GP_greater_than_0').mean()
data_var = X_train.groupby('GP_greater_than_0').var()
gaussian_columns = X_train.columns.drop(['country_group', 'Position', 'GP_greater_than_0'])


def bessel_var(gaussian_columns):
    df = pd.DataFrame(index=['no', 'yes'], columns=gaussian_columns)
    for i in gaussian_columns:
        s_yes = X_train[i][X_train.GP_greater_than_0 == 'yes'] - X_train[i][X_train.GP_greater_than_0 == 'yes'].mean()
        s_no = X_train[i][X_train.GP_greater_than_0 == 'no'] - X_train[i][X_train.GP_greater_than_0 == 'no'].mean()
        df[i]['no'] = compute_bessel(s_no)
        df[i]['yes'] = compute_bessel(s_yes)
    return df


def compute_bessel(age_norm):
    summation = 0
    for i in age_norm:
        summation += i ** 2
    s_sqr = summation / (age_norm.count() - 1)
    return s_sqr


df = bessel_var(gaussian_columns)


def mean(feature, GP):
    if (GP == 'no'):
        GP = 0
    else:
        GP = 1
    return data_mean[feature][GP]


def variance(feature, GP):
    if (GP == 'no'):
        GP = 0
    else:
        GP = 1
    return data_var[feature][GP]


def get_bessel(feature, GP):
    if (GP == 'no'):
        GP = 0
    else:
        GP = 1
    return df[feature][GP]


def gaussian(x, mean_val, var_val):
    p = (1 / (np.sqrt(2 * 3.14 * var_val)) * (math.exp((-(x - mean_val) ** 2) / (2 * var_val))))
    return p


def likelihood_gauss(test_set, row_index, GP_grt):
    x = 1
    iter_list = test_set.columns.drop(['country_group', 'Position'])
    for i in list(iter_list):
        x *= gaussian(test_set[i][row_index], mean(i, GP_grt), variance(i, GP_grt))
    return x


def likelihood_bessel(test_set, row_index, GP_grt):
    y = 1
    iter_list = test_set.columns.drop(['country_group', 'Position'])
    for i in list(iter_list):
        y *= gaussian(test_set[i][row_index], mean(i, GP_grt), get_bessel(i, GP_grt))
    return y


def classifier_func(test_set, test_name):
    list_pred = []

    for i in test_set.index:
        P_country_GP_grt = X_train['country_group'][(X_train.country_group == test_set['country_group'][i]) & (
        X_train.GP_greater_than_0 == 'yes')].count() / num_GPgrt_0
        P_country_GP_not = X_train['country_group'][(X_train.country_group == test_set['country_group'][i]) & (
        X_train.GP_greater_than_0 == 'no')].count() / num_GP_not_grt_0
        P_pos_GP_grt = X_train['Position'][(X_train.Position == test_set['Position'][i]) & (
        X_train.GP_greater_than_0 == 'yes')].count() / num_GPgrt_0
        P_pos_GP_not = X_train['Position'][(X_train.Position == test_set['Position'][i]) & (
        X_train.GP_greater_than_0 == 'no')].count() / num_GP_not_grt_0

        if (test_name == 'gaussian'):
            no_prob = P_num_not_grt_0 * P_country_GP_not * P_pos_GP_not * likelihood_gauss(test_set, i, 'no')
            yes_prob = P_num_grt_0 * P_country_GP_grt * P_pos_GP_grt * likelihood_gauss(test_set, i, 'yes')

            if (no_prob > yes_prob):
                list_pred.append('no')
            else:
                list_pred.append('yes')
        elif (test_name == 'bessel'):
            no_prob_bessel = P_num_not_grt_0 * P_country_GP_not * P_pos_GP_not * likelihood_bessel(test_set, i, 'no')
            yes_prob_bessel = P_num_grt_0 * P_country_GP_grt * P_pos_GP_grt * likelihood_bessel(test_set, i, 'yes')
            if (no_prob_bessel > yes_prob_bessel):
                list_pred.append('no')
            else:
                list_pred.append('yes')

    return list_pred

list_get_gaussian = classifier_func(X_test, 'gaussian')
list_get_bessel = classifier_func(X_test, 'bessel')
gauss_matrix = np.array(list_get_gaussian)
bessel_matrix = np.array(list_get_bessel)

from sklearn.metrics import confusion_matrix

cm_gauss = confusion_matrix(y_test, gauss_matrix)
cm_bessel = confusion_matrix(y_test, bessel_matrix)
print('Gaussian Confusion matrix :')
print(cm_gauss)
print('Bessel Confusion matrix :')
print(cm_bessel)
