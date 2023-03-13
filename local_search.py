import random
import numpy as np
from bisect import bisect_left,bisect_right
from deap import base
toolbox = base.Toolbox()
from fitness_calculation import fit_train1
from encoding import find_selected_feature_index


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


def calculate_mic_pop_new(pop,space,all_pre_intervals,mic_value):
    pop_mic = []
    for p in pop:
       x = find_selected_feature_index(p, space, all_pre_intervals)
       x = list(set(x))
       mic_value_temp = [mic_value[t] for t in x]
       pop_mic.append(np.mean(mic_value_temp))
    return pop_mic


def get_all_index(lst,item):
    tmp = []
    tag = 0
    for i in lst:
        if i == item:
            tmp.append(tag)
        tag +=1
    return tmp



def random_select(space,num):
    temp = 0
    used_space =[]
    for i in range(len(num)):
        temp = temp + num[i]
        # end_point = temp
        # start_point = end_point - num[i]
        # temp_selected_feature = x[start_point:end_point]
        # while -1 in temp_selected_feature:
        #   temp_selected_feature.remove(-1)
        # print('real selected features',temp_selected_feature)
        space_temp = space[temp-1]
        used_space.append(space_temp)
    random_choose_one_space = random.sample(used_space,1)
    random_choose_one_space = random_choose_one_space[0]
    corresponding_dimension_solution = []
    for j in range(len(space)):
        if space[j] == random_choose_one_space:
            corresponding_dimension_solution.append(j)
    # print(corresponding_dimension_solution)
    return corresponding_dimension_solution,random_choose_one_space




def neighhood_move_up(solution,x,chosen_dimensions,chosen_space,all_pre_intervals):##move up
    member = toolbox.clone(solution)
    corresponding_selected_features = [x[i] for i in chosen_dimensions]
    position_index = []###return the index of the selected features in its corresponding space
    if -1 in corresponding_selected_features:##some dimension didn't select features
        index_whole= get_all_index( corresponding_selected_features,-1)
        index_minus_1 = random.sample(index_whole,1)
        index_minus_1 = index_minus_1[0]
        move_up_position = chosen_dimensions[index_minus_1]
        while -1 in corresponding_selected_features:##remove the dimension of no feature is selected
              corresponding_selected_features.remove(-1)
        for i in corresponding_selected_features:
            position_index.append(chosen_space.index(i))
    else:##select the dimension with most unimportant feature
        for i in corresponding_selected_features:
                position_index.append(chosen_space.index(i))
        minimal_index = position_index.index(min(position_index))####the most unimportant feature
        # print(minimal_index)##this is kind of the soting of the dimension
        move_up_position = chosen_dimensions[minimal_index]#the dimension of solution will to change
    if corresponding_selected_features == []:###no feature is selected
         index_change_random = random.sample(chosen_dimensions,1)
         move_up_position = index_change_random[0]
         temp = all_pre_intervals[move_up_position][-2:]
         random_number = random.uniform(temp[-2], temp[-1])
         member[move_up_position] = random_number
         return member, move_up_position
    else:
    ###to determine to change as who
        temp = len(chosen_space) - max(position_index) - 1
        if temp == 0:##that means the most important feature is unselected
            member = member
            move_up_position = []
            return member,move_up_position
        else:##that means the most important feature is unselected
            temp = all_pre_intervals[move_up_position][-2:]
            random_number = random.uniform(temp[-2],temp[-1])
            member[move_up_position] = random_number
            return member,move_up_position



def neighhood_remove(ref,x, all_pre_intervals,move_up_position,mic_value):
    member = toolbox.clone(ref)
    mic_x = []
    for j in x:
        if j >= 0:
                mic_x.append(mic_value[j])
        else:
                mic_x.append(10000)###no feature is selected and then use a maximal number
    index_sort = np.argsort(mic_x)#remove the feature with the minimal mic value
    if move_up_position==[]:
        index = index_sort[0]
    else:
        index = (index_sort[0] == move_up_position)*index_sort[1] + (index_sort[0] != move_up_position)*index_sort[0]
    temp = all_pre_intervals[index][0:2]
    random_number = random.uniform(temp[0],temp[1])
    member[index] = random_number
    return member






def neighhood_minus_1(solution,mic_solution,x,space,all_pre_intervals,mic_value,index_all_minus1,random_dimension):
    nei3 = toolbox.clone(solution)
    all_features = space[random_dimension]
    space_index = get_all_index(space, all_features)  ##the dimensions corresponding the same interval
    se_features = [x[ii] for ii in space_index]
    while -1 in se_features:  ##remove the dimension of no feature is selected
        se_features.remove(-1)
    remaining_features = list(set(all_features) - set(se_features))
    # print(x,index_all_minus1)
    # print('random_dimension',random_dimension,all_mic[i])
    # print('space',space_index,all_features,se_features,remaining_features)
    mic_re = [mic_value[jj] for jj in remaining_features]
    used_index = np.argwhere(mic_re > mic_solution)
    if len(used_index) > 0:
        random_one = random.sample(list(used_index), 1)  ###the selected dimension
        add_one = random_one[0][0]
        position = np.argwhere(all_features == remaining_features[add_one])
        position = position[0][0]
        ref_di = all_pre_intervals[random_dimension]
        nei3[random_dimension] = nei3[random_dimension] + 0.00000000000001 + (position + 1) * (ref_di[1] - ref_di[0])
    return nei3,used_index




#the one: one feature is used twice%%%%%%%%%%%%%%%%%%%plus/minus the interval, modify all of them
#the two: all dimensions select features%%%%%%%%%%%%remove one feature
def add_remove_local_search(new,space,all_pre_intervals,x_train,y,mic_value,fit_count,num):####the surrogate x_train
    pop_new = toolbox.clone(new)
    all_mic = calculate_mic_pop_new(new, space, all_pre_intervals, mic_value)
    pop_fit = [ind.fitness.value for ind in new]
    for i in range(len(new)):
        mic_solution = all_mic[i]
        x = find_selected_index(new[i],space,all_pre_intervals)
        # index_all_minus1 = get_all_index(x,-1)
        x_useful = toolbox.clone(x)
        while -1 in x_useful:  ##remove the dimension of no feature is selected
            x_useful.remove(-1)
        unique_x = list(set(x_useful))
        index_temp = []
        for in1 in unique_x:
            id1 = [j1 for j1, j2 in enumerate(x) if j2 == in1]
            index_temp.append(id1)###find the group with the same selected features
        index2 = [j1 for j1, j2 in enumerate(index_temp) if len(j2) > 1]
        ###[3] ##########[[10], [0], [4], [1, 2,4], [9], [7]]
        ###[0,1] ##########[[1,2,3], [0,5], [4], [6], [8], [9], [7]]
        if len(index2)>0:####that may have multiple groups
            for id2 in index2:
                random_dimension = index_temp[id2][0]##used for collecting features
                nei1 = toolbox.clone(new[i])
                all_features = space[random_dimension]
                space_index = get_all_index(space, all_features)  ##the dimensions corresponding the same interval
                se_features = [x[ii] for ii in space_index]
                while -1 in se_features:  ##remove the dimension of no feature is selected
                    se_features.remove(-1)
                remaining_features = list(set(all_features) - set(se_features))
                # print(x)
                # print('random_dimension',random_dimension,all_mic[i])
                # print('space',space_index,all_features,se_features,remaining_features)
                all_changed_dimension = random.sample(index_temp[id2],len(index_temp[id2])-1)
                mic_re = [mic_value[jj] for jj in remaining_features]
                used_index = np.argwhere(mic_re >= mic_solution)
                # print(all_changed_dimension)
                if len(used_index) >= len(all_changed_dimension):##use mic to choose solutions
                    random_one = random.sample(list(used_index), len(all_changed_dimension))  ###the selected dimension
                    for iii in range(len(random_one)):
                        add_one = random_one[iii][0]
                        changed_position = all_changed_dimension[iii]
                        position = np.argwhere(all_features == remaining_features[add_one])
                        position = position[0][0]
                        ref_di = all_pre_intervals[random_dimension]
                        nei1[changed_position] = random.uniform((position + 1) * (ref_di[1] - ref_di[0]),
                                                                (position + 2) * (ref_di[1] - ref_di[0]))
                else:###otherwise randomly choose features from the remaining ones
                    random_one = random.sample(remaining_features, len(all_changed_dimension))  ###the selected dimension
                    for iii in range(len(random_one)):
                        add_one = random_one[iii]
                        changed_position = all_changed_dimension[iii]
                        position = np.argwhere(all_features == add_one)
                        position = position[0][0]
                        ref_di = all_pre_intervals[random_dimension]
                        nei1[changed_position] = random.uniform((position + 1) * (ref_di[1] - ref_di[0]),
                                                                (position + 2) * (ref_di[1] - ref_di[0]))
            ###first one: didn't use removing operator:
            nb1_sf = fit_train1(nei1, x_train, y, space, all_pre_intervals)
            fit_count = fit_count + 1
            if nb1_sf < pop_fit[i]:
                pop_new[i] = nei1
                pop_fit[i] = nb1_sf
            size = find_selected_feature_index(pop_new[i], space, all_pre_intervals)
            if len(size) > 1:
                move_up_position = []
                nei2 = neighhood_remove(pop_new[i], x, all_pre_intervals, move_up_position, mic_value)
                nb2_sf = fit_train1(nei2, x_train, y, space, all_pre_intervals)
                fit_count = fit_count + 1
                if nb2_sf <= pop_fit[i]:
                    pop_new[i] = nei2
                    pop_fit[i] = nb2_sf
        else:######no feature is used twice, then use move_up_operator
            # mic_each_space = []
            # for ss in space:
            #     min_temp = [mic_value[sss] for sss in ss]
            #     mic_each_space.append(np.mean(min_temp))
            # chosen_space = np.argsort(-np.array(mic_each_space))
            # chosen_space = chosen_space[0]
            nei3 = toolbox.clone(new[i])
            space_length = [len(s) for s in space]####[8,8,7,7,3,2]
            certain_length = [j1 for j1, j2 in enumerate(space_length) if j2 > np.max(space_length)/2]##[0,1,2,3]
            index_pace = random.sample(certain_length,1)#[3]
            all_features = space[index_pace[0]]
            ref_di = all_pre_intervals[index_pace[0]]
            index_space = get_all_index(space_length,space_length[index_pace[0]])##[2,3]
            se_features = [x[ii] for ii in index_space] ##[15,-1]
            # print(x)
            # print(se_features,all_features)
            if -1 in se_features:###have feature unselected
                change_position = get_all_index(se_features, -1)  ##[0,1]
                changed_position = index_space[change_position[0]]
            else:
                position1 = [get_all_index(all_features,f_s) for f_s in se_features]
                position1 = [ss[0] for ss in position1]
                sssss = np.argsort(position1)
                changed_position = index_space[sssss[0]]
            while -1 in se_features:  ##remove the dimension of no feature is selected
                se_features.remove(-1)
            remaining_features = list(set(all_features) - set(se_features))
            # print('space',all_features,se_features,remaining_features)
            mic_re = [mic_value[jj] for jj in remaining_features]
            used_index = np.argwhere(mic_re >= mic_solution)
            # print('used_index',used_index)
            if len(used_index) !=0:  ##use mic to choose solutions
                random_one = random.sample(list(used_index), 1)  ###the selected dimension
                # print('random_one', random_one,random_one[0])
                # print(remaining_features)
                add_one = random_one[0][0]
                position = np.argwhere(all_features == remaining_features[add_one])
                position = position[0][0]
                nei3[changed_position] = random.uniform((position + 1) * (ref_di[1] - ref_di[0]),
                                                            (position + 2) * (ref_di[1] - ref_di[0]))
            elif len(used_index)==0 and len(remaining_features) != 0:
                random_one = random.sample(remaining_features, 1)  ###the selected dimension
                for iii in range(len(random_one)):
                    add_one = random_one[iii]
                    position = np.argwhere(all_features == add_one)
                    position = position[0][0]
                    nei3[changed_position] = random.uniform((position + 1) * (ref_di[1] - ref_di[0]),
                                                            (position + 2) * (ref_di[1] - ref_di[0]))
            nb3_sf = fit_train1(nei3, x_train, y, space, all_pre_intervals)
            fit_count = fit_count + 1
            if nb3_sf < pop_fit[i]:
                pop_new[i] = nei3
                pop_fit[i] = nb3_sf
            size1 = find_selected_feature_index(pop_new[i], space, all_pre_intervals)
            if len(size1) > 1:
                move_up_position = []
                nei2 = neighhood_remove(new[i],x,all_pre_intervals, move_up_position, mic_value)
                nb2_sf = fit_train1(nei2, x_train, y, space, all_pre_intervals)
                fit_count = fit_count + 1
                if nb2_sf <= pop_fit[i]:
                  pop_new[i] = nei2
                  pop_fit[i] = nb2_sf
    return pop_new,pop_fit,fit_count
