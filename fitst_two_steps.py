from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=UserWarning)
import numpy as np
from minepy import MINE
mine = MINE(alpha=0.6, c=15, est="mic_approx")



def two_steps(mic_value,training_data):
  #####################################first step: get rou0
    rou0 = 0.15 * max(mic_value)
  #####################################remove features
    removed_index = np.argwhere(mic_value<= rou0)
    saved_index = np.argwhere(mic_value> rou0)
    # print(removed_index)
    if len(removed_index) == 0:####this one means that no features are removed from the whole
        index, num = cluster_features_song(mic_value,training_data)
        save_space_to_indi = []
        for i in range(len(index)):
            temp = [index[i]] * num[i]
            save_space_to_indi.extend(temp)
    else:##this one means that some features with low correlation measure values are removed
        removed_index = [i[0] for i in removed_index]
        saved_index = [i[0] for i in saved_index]
        index, num = cluster_features_song_removed(saved_index,mic_value,training_data)
        save_space_to_indi = []
        for i in range(len(index)):
            temp = [index[i]] * num[i]
            save_space_to_indi.extend(temp)
    save_space_to_indi = [tt[::-1] for tt in save_space_to_indi]
    return save_space_to_indi,num


def cluster_features_song(mic_value,data):####this means the first step remove 0 feature
    clusters = {}
    inde = np.argsort(-mic_value) ###get the index of sorting, from max to min
    inde = list(inde)
    t = 0
    while len(inde) != 0:
       temp = []
       temp.append(inde[0])  ###the most important one is in the first
       for i in range(1,len(inde)):
          mine.compute_score(data[:, inde[i]], data[:, inde[0]])
          value = mine.mic()
          if value >= min(mic_value[inde[0]], mic_value[inde[i]]):
             temp.append(inde[i])
       clusters[t] = temp
       for j in temp:
            inde.remove(j)###remove element based on the value
       t = t+1
    num = [int(np.sqrt(len(clusters[j]))) for j in range(len(clusters))]#######maximom number of features selected from each cluster
    return clusters,num



def cluster_features_song_removed(saved_index,mic_value,data):
    saved_index = np.array(saved_index)#####the dimension after the first step
    after_first_step_mic = [mic_value[i] for i in saved_index]
    clusters = {}
    inde = np.argsort(-np.array(after_first_step_mic))  ###get the index of sorting, from max to min
    inde = list(inde)
    t = 0
    while len(inde) != 0:
        temp = []
        temp.append(inde[0])  ###the most important one is in the first
        for i in range(1, len(inde)):
            mine.compute_score(data[:,saved_index[inde[i]]], data[:,saved_index[inde[0]]])###from the original data, need to use original index
            value = mine.mic()
            if value >= min(after_first_step_mic[inde[0]], after_first_step_mic[inde[i]]):###after_first_step_mic includes saved_index
                temp.append(inde[i])
        clusters[t] = temp
        for j in temp:
            inde.remove(j)  ###remove element based on the value
        t = t + 1
    num = []
    for j in range(len(clusters)):
       temp1 = saved_index[clusters[j]]
       clusters[j] = list(temp1)
       num.append(int(np.sqrt(len(clusters[j]))) ) #######maximom number of features selected from each cluster
    return clusters, num

def niche_introduce(dis,ii):
    ss_i = list(np.argsort(dis[ii, :]))  ###the index in the whole population from min to max
    niche_index = ss_i[:4]  #####the three having the smallest distance to the current individual, the first one is itself
    three_distance = [dis[ii, iii] for iii in niche_index[1:4]]
    gaussion_mean = np.mean(three_distance)
    gaussion_std = np.std(three_distance)
    range_low = gaussion_mean - 3 * gaussion_std
    range_high = gaussion_mean + 3 * gaussion_std
    for j in ss_i:
        if range_low <= dis[ii, j] <= range_high:
            niche_index.append(j)
    niche_index = list(sorted(set(niche_index)))###sort is to ensure the first ndividual is itself
    return niche_index
