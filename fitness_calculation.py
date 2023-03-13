from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=UserWarning)
import array
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from itertools import chain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from encoding import find_selected_feature_index
import operator
from sklearn.metrics import balanced_accuracy_score


def fit_train1(x1,train_data,label,index,all_pre_intervals):
    value_position = find_selected_feature_index(x1,index,all_pre_intervals)
    if len(value_position) == 0:
        f1 = 10000
        f2 = 0
    else:
      tr = train_data[:, value_position]
      clf = KNeighborsClassifier(n_neighbors = 5)
      pre_label = cross_val_predict(clf, tr,label, cv=5)
      scores = balanced_accuracy_score(label, pre_label)
      #scores = cross_val_score(clf, tr,label, cv = 5)
      f1 = np.mean(1 - scores)
      f2 = len(value_position)
    f = f1 + 0.000001 * f2
    return f


def evaluate_test_data(x1, train_data, test_data,index,all_pre_intervals):
    train, test = train_data[:,1:], test_data[:,1:]
    training_label = train_data[:,0]
    test_label = test_data[:,0]
    value_position = find_selected_feature_index(x1,index,all_pre_intervals)
    tr = train[:, value_position]  #####training data with selected features
    te = test[:, value_position]#####testing data with selected features
    wrong = 0
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(tr, training_label)
    outputLabel = clf.predict(te)
    for i in range(len(te)):
        if outputLabel[i] != test_label[i]:
            wrong = wrong + 1
    f1 = wrong/len(test_label)
    return f1
