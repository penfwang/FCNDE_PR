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
from minepy import MINE
import DE_operator
from fitst_two_steps import two_steps,niche_introduce
from fitness_calculation import fit_train1,evaluate_test_data
from deal_with_duplication import delete_duplicate,modify_duplication
from encoding import get_all_intervals,initialization
import sys,saveFile
import math,time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import local_search


def uniform(low, up, size=None):####generate a matrix of the range of variables
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]



def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result


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


def euclidean_distance(x1,x2):
    s1 = toolbox.clone(x1)
    s2 = toolbox.clone(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    temp = sum((s1-s2)**2)
    temp1 = np.sqrt(temp)
    return temp1



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))####minimise two objectives
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def main(seed,dataset_name):
    random.seed(seed)
    folder1 = '/nfs/home/wangpe/split_73' + '/' + 'train' + str(dataset_name) + ".npy"
    x_train = np.load(folder1)
    class_n = set(x_train[:, 0])
    T = len(x_train) - 1
    ee = 1 / (T * len(class_n))
    if ee > 0.0001:
       ee = 0.0001
    mine = MINE(alpha=0.6,c=15,est="mic_approx")
    NGEN = 100  ###the number of generation
    NDIM = x_train.shape[1] - 1####the length of individualMU = NDIM
    if NDIM < 200:
        MU = NDIM ####the number of particle
    else:
        MU = 200  #####bound to 200
    Max_FES = MU * NGEN
    #training_data = normalise(x_train[:, 1:])
    training_data = x_train[:, 1:]
    #training_data = preprocessing.normalize(x_train[:, 1:])
    #crr = kNNCRR()
    #X_crr, y_crr = crr.fit(x_train[:, 1:], x_train[:, 0])
    mic_value = []
    for i_in in range(NDIM):  ####just know the dimensinal number
      mine.compute_score(training_data[:,i_in], x_train[:,0])
      mic_value.append(mine.mic())
    mic_value = np.array(mic_value)
    ##############################first two steps
    save_space_to_indi,num = two_steps(mic_value,training_data)###num means the maximum number of features all selected
    ##############################first two steps
    all_pre_intervals  = get_all_intervals(save_space_to_indi)
    BOUND_LOW, BOUND_UP = 0.0, 1.0#####
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, sum(num))  #####dertemine the way of randomly generation and gunrantuu the range
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)  ###fitness
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ##particles
    toolbox.register("select", DE_operator.selNS)  ##from min to max
    toolbox.register("select1", DE_operator.selection_compared_with_nearest)
    min_fitness = []
    unique_number = []
    offspring = toolbox.population(n=MU)
    #offspring = initialization(offspring,save_space_to_indi,all_pre_intervals,mic_value)
    
    toolbox.register("evaluate", fit_train1, train_data=training_data,label=x_train[:,0],index=save_space_to_indi,all_pre_intervals = all_pre_intervals)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  #####toolbox.evaluate = fit_train
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.value = fit
    pop_fit = [ind.fitness.value for ind in offspring]
    min_fitness.append(min(pop_fit))
    fit_num = len(offspring)
    pop_surrogate = delete_duplicate(offspring,save_space_to_indi,all_pre_intervals)
    unique_number.append(len(pop_surrogate))
    dis = np.zeros((MU, MU))
    for i in range(MU):
        for j in range(MU):
            #pca = PCA(1)
            #pca.fit(offspring)
            #offspring_new_space = pca.transform(offspring)
            #dis[i, j] = euclidean_distance(offspring_new_space[i], offspring_new_space[j])
            dis[i, j] = euclidean_distance(offspring[i], offspring[j])
    for gen in range(1, NGEN):
        pop_new = toolbox.clone(offspring)
        for ii in range(len(offspring)):  #####upate the whole population
            niche_index = niche_introduce(dis,ii)
            y_new,nbest= DE_operator.mutDE_LBP_NGI(offspring,niche_index,ii,0.5)
            for i_z in range(len(y_new)):
                if y_new[i_z] > 1:
                    y_new[i_z] = 1
                if y_new[i_z] < 0:
                    y_new[i_z] = 0
            pop_new[ii] = DE_operator.cxBinomial(offspring[ii], y_new, 0.5)  ###crossover
            del pop_new[ii].fitness.values  ###delete the fitness
        ##################################################
        pop_new = modify_duplication(offspring,pop_new, save_space_to_indi,all_pre_intervals)
        for i_z1 in range(len(pop_new)):
          for i_z2 in range(len(pop_new[0])):
            if pop_new[i_z1][i_z2] > 1:
                pop_new[i_z1][i_z2] = 1
            if pop_new[i_z1][i_z2] < 0:
                pop_new[i_z1][i_z2] = 0
        #############################################################new
        invalid_ind = [ind for ind in pop_new if not ind.fitness.valid]
        fitne = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit1 in zip(invalid_ind, fitne):
            ind.fitness.value = fit1
        fit_num = fit_num + len(offspring)
        #############################################################new
        pop_mi = pop_new + offspring
        pop1 = delete_duplicate(pop_mi,save_space_to_indi,all_pre_intervals)
        offspring = toolbox.select(pop1, MU, ee)###compare with the current
        
        #offspring = toolbox.select1(offspring, pop_new)
        pop_fit = [ind.fitness.value for ind in offspring]  ######selection from author
        min_fitness.append(min(pop_fit))
        if gen > 500 and np.std(min_fitness[-5:])==0:
            offspring_temp = [t for t in offspring]
            print('designed case',min_fitness[-2:],np.std(min_fitness[-2:]))
            index_mulmodal = np.argwhere(abs(pop_fit-min(pop_fit)) <= ee)
            index_mulmodal = [m[0] for m in index_mulmodal]
            pop_part = [offspring[m1] for m1 in index_mulmodal]
            for m2 in index_mulmodal:
                offspring_temp.remove(offspring[m2])
            pop_part,new_pop_fit,fit_num=local_search.add_remove_local_search(
                pop_part,save_space_to_indi,all_pre_intervals,training_data,x_train[:,0],mic_value,fit_num,num)
            invalid_ind = [ind for ind in pop_part if not ind.fitness.valid]
            for ind, fit1 in zip(invalid_ind, new_pop_fit):
                ind.fitness.value = fit1
            offspring_temp.extend(pop_part)
            offspring = [s for s in offspring_temp]
        pop_surrogate.extend(delete_duplicate(pop_new,save_space_to_indi,all_pre_intervals))
        pop_surrogate = delete_duplicate(pop_surrogate,save_space_to_indi,all_pre_intervals)
        unique_number.append(len(pop_surrogate))
        for i in range(MU):
            for j in range(MU):
                #pca = PCA(1)
                #pca.fit(offspring)
                #offspring_new_space = pca.transform(offspring)
                #dis[i, j] = euclidean_distance(offspring_new_space[i], offspring_new_space[j])
                dis[i, j] = euclidean_distance(offspring[i], offspring[j])
        if fit_num > Max_FES:
            break
    return offspring, min_fitness,save_space_to_indi,all_pre_intervals,unique_number




if __name__ == "__main__":
    dataset_name = str(sys.argv[1])
    seed = str(sys.argv[2])
    start = time.time()
    pop, min_fitness, save_space_to_indi, all_pre_intervals,unique_number = main(seed, dataset_name)
    end = time.time()
    running_time = end - start
    pop1 = delete_duplicate(pop,save_space_to_indi,all_pre_intervals)
    pop_fit = [ind.fitness.value for ind in pop1]
    EXA_array = np.array(pop1)
    saveFile.saveAllfeature2(seed, dataset_name, EXA_array)
    saveFile.saveAllfeature3(seed, dataset_name, pop_fit)
    saveFile.saveAllfeature5(seed, dataset_name, unique_number)
    saveFile.saveAllfeature6(seed, dataset_name, min_fitness)
    saveFile.saveAllfeature7(seed, dataset_name, running_time)
    #saveFile.saveAllfeature9(seed, dataset_name, all_pre_intervals)
    #saveFile.saveAllfeature10(seed, dataset_name, save_space_to_indi)
