import random
import numpy as np
from bisect import bisect_left,bisect_right


def initialization(offspring,save_space_to_indi,all_pre_intervals,mic_value):
    for i in offspring:
          for j in range(len(i)):
              part_mic = [mic_value[m] for m in save_space_to_indi[j]]
              p = np.max(part_mic)/np.max(mic_value)
              if random.uniform(0,1) < p:
                  i[j] = random.uniform(all_pre_intervals[j][1], 1)
              else:
                  i[j] = random.uniform(0, all_pre_intervals[j][1])
              # i[j] = random.uniform(1.0, 2.0)
    return offspring


def limit_range(y,index):
    for i_z in range(len(y)):
        max_value = len(index[i_z])+1
        if y[i_z] < 0:
            y[i_z] = 0
        if y[i_z] > max_value:
            y[i_z] = max_value
    return y




def get_all_intervals(index):
    all_pre_intervals = []
    length = [len(i)+1 for i in index]#####the real features in each cluster plus select 0 feature
    for i in range(len(index)):
        pre = []  ####the produced intervals to determine whicch feature is selected
        for j in range(length[i] + 1):
            pre.append(j)######produce number from 0 to maximum value
        pre = np.array(pre)/np.max(pre)######normilize the intervals
        all_pre_intervals.append(pre)
    return all_pre_intervals



def find_selected_feature_index(x1,index,all_intervals):############employing bach's encoding
    value_position = []###the index of selected features
    # all_intervals = get_all_intervals(index)
    for x in range(len(x1)):
        pre = all_intervals[x]
        candidate_feature_index = index[x]####the x-dimension related to all the features
        temp = int(bisect_left(pre,x1[x]))##the x-dimension's interval position
        if temp > 1:
            # print('sssss',x1[x],temp,candidate_feature_index,pre,candidate_feature_index[temp-2])
            # print('sssss2', x1[x], temp, len(candidate_feature_index), len(pre), candidate_feature_index[temp - 2])
            selected_feature_index = candidate_feature_index[temp-2]
            value_position.append(selected_feature_index)
    # exit()
    value_position = list(set(value_position))
    return value_position
