####song's idea
##A Fast Hybrid Feature Selection Based on Correlation-Guided Clustering and Particle Swarm Optimization for High-Dimensional Data
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

def normalise_whole(data,test_data):
    for i in range(data.shape[1]):
      column = data[:,i]
      mini = min(column)
      maxi = max(column)
      if mini != maxi:
         for j in range(len(column)):
            data[j,i] = (column[j] - mini) / (maxi - mini)
         test_column = test_data[:, i]
         for jj in range(len(test_column)):
           test_data[jj,i] = (test_column[i] - mini) / (maxi - mini)
    return data,test_data


def normalise(data):
    data2 = np.ones(((data.shape[0]),data.shape[1]))
    # print(data.shape[0],data.shape[1])
    for i in range(data.shape[1]):
      column = data[:,i]
      mini = min(column)
      maxi = max(column)
      for j in range(len(column)):
         if mini != maxi:
           data2[j,i] = (column[j] - mini) / (maxi - mini)
         else:
           data2[j, i] = 1
    return data2


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
    #train, test = normalise_whole(train_data[:,1:], test_data[:,1:])
    train, test = train_data[:,1:], test_data[:,1:]
    #train, test =normalise(train_data[:,1:]), normalise(test_data[:,1:])
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


def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   # shape[0]表示行数
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = squaredDiff.sum(axis = 1)   # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sorted_ClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_ClassCount[0][0]
