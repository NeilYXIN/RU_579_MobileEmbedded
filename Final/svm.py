import numpy as np
import scipy.io
import os
import math
import random
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import svm
from sklearn import metrics

# K-fold cross validation, K=num_Fold
def cross_validation(X, y, num_Fold): 
    # Define the split - into k folds 
    kf = StratifiedKFold(n_splits=num_Fold, shuffle=False);

    # a list that stores the accuracy of each fold to calculate overall accuracy
    accus=[];

    # Predefine the sum of all confusion matrix
    big_conf_mat=np.zeros((len(set(y)),len(set(y)))); 
    # Calculate the accuracy for each fold
    for train_index, test_index in kf.split(X,y):
        # train/test split
        train_index=np.array(train_index);
        test_index=np.array(test_index);
        X=np.array(X);
        y=np.array(y);
        X_train, X_test = X[train_index], X[test_index];
        y_train, y_test = y[train_index], y[test_index];

        # Fit the SVM model, kernel function: linear
        clf = svm.SVC(kernel='linear').fit(X_train, y_train);
        # Predict on the test data
        y_pred = clf.predict(X_test);

        # Generate confusion matrix for this fold
        conf_mat=confusion_matrix(y_test, y_pred, );
        print(conf_mat);

        # Sum the overall confusion matrix
        big_conf_mat=big_conf_mat+conf_mat;

        # Calculate the accuracy and append to the list
        accus.append(metrics.accuracy_score(y_test, y_pred));
        print("Accuracy:",accus[-1],'\n');
    
    # Print the sum of each fold's confusion matrix
    print('SumConfMat:');
    print(big_conf_mat);
    # Print the overall accuracy
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (np.mean(accus), np.std(accus) * 2),'\n');


"""
# How to use SVM
# Dataset:
    X: m*n matrix representing the feature, m: number of data samples, n: number of features
    y: 1*m vector representing the label, m: number of data samples

clf = svm.SVC(kernel='linear').fit(X_train, y_train); # Train training set using svm
y_pred = clf.predict(X_test);
conf_mat=confusion_matrix(y_test, y_pred, );
accu=metrics.accuracy_score(y_test, y_pred);
print("Accuracy:",accu,'\n')

A link that introduces how to save the svm model to file and load the model from local file:
https://scikit-learn.org/stable/modules/model_persistence.html
"""