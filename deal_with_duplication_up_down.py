from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=UserWarning)
import random
import numpy as np
from fitness_calculation import find_selected_feature_index
from bisect import bisect_left,bisect_right



def findindex(org, x):
    result = []
    for k,v in enumerate(org): #
        if v == x:
            result.append(k)
    return result

def get_all_index(lst,item):
    tmp = []
    tag = 0
    for i in lst:
        if i == item:
            tmp.append(tag)
        tag +=1
    return tmp


def more_confidence(EXA, index_of_objectives):#####this is for the original confidence calculation using threshold o
    cr = np.zeros((len(index_of_objectives),1))
    for i in range(len(index_of_objectives)):###the number of indexes
        object = EXA[index_of_objectives[i]]
        for ii in range(len(object)):###the number of features
               cr[i,0] = cr[i,0] + object[ii]
    sorting = np.argsort(cr[:,0])####sorting from maximum to minimum
    index_one = index_of_objectives[sorting[0]]
    return index_one

def find_selected_index(x1,index,all_intervals):############employing bach's encoding
    value_position = []###the index of selected features
    # all_intervals = get_all_intervals(index)
    for x in range(len(x1)):
        pre = all_intervals[x]
        candidate_feature_index = index[x]####the x-dimension related to all the features
        temp = int(bisect_left(pre,x1[x]))##the x-dimension's interval position
        if temp > 1:
            selected_feature_index = candidate_feature_index[temp-2]
            value_position.append(selected_feature_index)
        else:
            value_position.append(-1)####he selected index based on each dimension, if unselected add -1
    value_position = list(value_position)
    return value_position


def delete_duplicate(EXA,space,all_pre_intervals):####list
    EXA1 = []
    all_index = []
    # print(len(EXA))
    for i in EXA:
        x = find_selected_feature_index(i,space,all_pre_intervals)
        x = sorted(x)
        temp = "".join(map(str, x))
        all_index.append(temp)
    single_index = set(np.array(all_index))
    single_index = list(single_index)####translate it's form in order to following operating.
    for i1 in range(len(single_index)):
       index_of_objectives = findindex(all_index, single_index[i1])##find the index of each unique combination
       if len(index_of_objectives) == 1:####no duplicated solutions
          for i2 in range(len(index_of_objectives)):
             EXA1.append(EXA[index_of_objectives[i2]])
       else:####some combination have more than one solutions.
           index_one = more_confidence(EXA, index_of_objectives)
           EXA1.append(EXA[index_one])
    return EXA1



def modify_duplication(old,new,space,all_pre_intervals):####list
    EXA1 = []
    all_index = []
    # print(len(EXA))
    for i in old:
        x = find_selected_feature_index(i,space,all_pre_intervals)
        x = list(set(x))
        x = sorted(x)
        temp = "".join(map(str, x))
        all_index.append(temp)
    single_index = set(np.array(all_index))
    single_index = list(single_index)####translate it's form in order to following operating.
    for j in range(len(new)):
        x1 = find_selected_feature_index(new[j],space,all_pre_intervals)
        x1 = sorted(x1)
        temp1 = "".join(map(str, x1))
        # print(single_index)
        index = get_all_index(single_index, temp1)
        if len(index) == 0:####some combination have never been shown before
              single_index.append(temp1)
        else:
            fixed_size = len(x1)
            interval_dimension = [m[1] - m[0] for m in all_pre_intervals]
            whole_index = find_selected_index(new[j], space, all_pre_intervals)
            # print(whole_index)
            if fixed_size == 1:####only one feature is selected
                for jj in range(len(interval_dimension)):
                   new[j][jj] = random.uniform(0,interval_dimension[jj])
                temp = random.randint(0, len(interval_dimension) - 1)
                new[j][temp] = new[j][temp] + interval_dimension[temp]
            else:
                index_l1 = [value1 for (value1,value) in enumerate(whole_index) if value !=-1]
                # print(index_l1)
                temp = random.sample(index_l1,fixed_size//2)
                ####need to check whether selected features have interval to increase
                whole_selected = sum([whole_index[t] for t in temp])
                largest_interval = sum([space[t][-1] for t in temp])
                if whole_selected != largest_interval:####have space to improve
                    for jj in temp:
                        new[j][jj] = new[j][jj] + interval_dimension[jj]
                else:##no space to improve
                    ss = np.argsort(temp)
                    delete_position = temp[ss[-1]]
                    l,u = all_pre_intervals[delete_position][0],all_pre_intervals[delete_position][1]
                    new[j][delete_position] = random.uniform(l,u)
    return new


def remove_duplicated_solutions_from_old_new(old,new,space,all_pre_intervals):
    unique_index = []
    duplicate_index = []
    all_index = []
    EXA = old
    for i in EXA:
        x = find_selected_feature_index(i,space,all_pre_intervals)
        x = list(set(x))
        print(x)
        x = sorted(x)
        temp = "".join(map(str, x))
        all_index.append(temp)
    single_index = set(np.array(all_index))
    single_index = list(single_index)####translate it's form in order to following operating.
    for i1 in range(len(new)):
        s = new[i1]
        x1 = find_selected_feature_index(s, space, all_pre_intervals)
        x1 = list(set(x1))
        x1 = sorted(x1)
        temp1 = "".join(map(str, x1))
        index_of_objectives = findindex( single_index, temp1)##find the index of each unique combination
        if len(index_of_objectives) == 0:####a new solution
            single_index.append(temp1)
            unique_index.append(i1)
        else:
           duplicate_index.append(i1)
    return unique_index, duplicate_index
