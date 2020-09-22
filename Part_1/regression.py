import sys
import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def check_matrix(array):
    """ Check matrix helper function """
    for i in range(len(array)):
        for j in range(len(array[i])):
            print(array[i][j], end=' ')
        print()
    print('\n')

def plot(x, y, x_label, y_label, title, savename):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x+b)
    plt.savefig((savename + '.png'))
    plt.figure()

"""Determine R^2 value"""
def r_squared(x, y):
    corr_mat = np.corrcoef(x, y)
    r2 = (corr_mat[0,1])**2
    return r2

"""Find the weight vector for the regression"""
def regress(x, y):
    """ W = (X^T X)^-1 X^T Y) """
    x_t = x.transpose()
    w = np.linalg.inv(x_t.dot(x)).dot(x_t).dot(y)
    return w

"""Use the weight vector to predict the y values"""
def predict(x, w):
    return x.dot(w)

def multiple_regression(train_x, test_x, train_y, test_y):
    """Add ones for constant in weight vector"""
    dummy = np.ones((len(train_x),1))
    train_x = np.append(dummy,train_x,axis=1)
    dummy = np.ones((len(test_x),1))
    test_x = np.append(dummy,test_x,axis=1)
    """Find weight vector"""
    w = regress(train_x, train_y)
    """apply weight vector to training and testing data"""
    p_train = predict(train_x, w)
    p_test = predict(test_x, w)
    train_r2 = r_squared(train_y, p_train)
    test_r2 = r_squared(test_y, p_test)
    return w, p_train, p_test, train_r2, test_r2

def mse(x, y, w):
    """ y = w1 * x1 + w2 * x2... """
    ase_X = np.power(np.subtract(y,x.dot(w)),2)
    ase = sum(ase_X)/len(ase_X)
    return ase

X = pd.read_csv('../data/3pt_train.csv',delimiter=",")
test_file = pd.read_csv('../data/3pt_test.csv', delimiter=",")
# print(X.head)

# slice the last column for Y
# Y = X[:,-1:]

# drop the invalid values and player names to convert to numpy
X = X.iloc[3:]
X = X.drop('player', 1)
test_X = test_file.drop('player', 1)
X = X.astype(float)
X = X.to_numpy().round(5)
test_X = test_X.astype(float)
test_X  = test_X.to_numpy().round(5)

cfg3_pct, cft_pct, cts_pct, fg3_pct = X[:, 4], X[:, 7], X[:, 8], X[:, 14]
test_fg3_pct = test_X[:, 14]

fg3_r2 = r_squared(cfg3_pct, fg3_pct)
print("Training College 3P% vs Pro 3P%", fg3_r2)

ft_r2 = r_squared(cft_pct, fg3_pct)
print("Training College FT% vs Pro 3P%", ft_r2)

t_fg3_r2 = r_squared(test_X[:, 4],test_fg3_pct )
print("Testing College 3P% vs Pro 3P%", t_fg3_r2)

t_ft_r2 = r_squared(test_X[:, 7],test_fg3_pct )
print("Testing FT% vs Pro 3P%", t_ft_r2)

plot(cfg3_pct, fg3_pct, "College 3P%", "Pro 3P%", "Training Data: College 3P% vs Pro 3P%", "college3_vs_pro3_train")
plot(cft_pct, fg3_pct, "College FT%", "Pro 3P%", "Training Data: College FT% vs Pro 3P%", "collegeFT_vs_pro3_train")

plot(test_X[:, 4], test_fg3_pct, "College 3P%", "Pro 3P%", "Testing Data: College 3P% vs Pro 3P%", "college3_vs_pro3_test")
plot(test_X[:, 7], test_fg3_pct, "College FT%", "Pro 3P%", "Testing Data: College FT% vs Pro 3P%", "collegeFT_vs_pro3_test")

train_1 = X[:, [4, 7]]
test_1 = test_X[:, [4, 7]]
w_1, p_train_1, p_test_1, r2_train_1, r2_test_1 = multiple_regression(train_1, test_1, fg3_pct, test_fg3_pct)

train_3 = X[:, [2, 3, 4, 5, 6, 7]]
test_3 = test_X[:, [2, 3, 4, 5, 6, 7]]
w_3, p_train_3, p_test_3, r2_train_3, r2_test_3 = multiple_regression(train_3, test_3, fg3_pct, test_fg3_pct)
print("test 3: cfg3, cfg3a, cfg3%, cft, cfta, cft%")
print("training r squared value: ", r2_train_3)
print("testing r squared value: ", r2_test_3)
print()

plot(p_test_3, test_fg3_pct, "Predicted 3P%", "Actual 3P%", "Regression Prediction vs Pro 3P%", "pred_vs_pro3")

names = test_file['player']
output_predictions = []
for i in range(len(p_test_3)):
    line = []
    line.append(names[i])
    line.append(p_test_3[i])
    line.append(test_fg3_pct[i])
    output_predictions.append(line)
    # print(line)


with open('output_predictions.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(output_predictions)
