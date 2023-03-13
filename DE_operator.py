from __future__ import division
import bisect
import math
import random
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict
import numpy as np
from deap import base


toolbox = base.Toolbox()

def euclidean_distance(x1,x2):
    s1 = toolbox.clone(x1)
    s2 = toolbox.clone(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    temp = sum((s1-s2)**2)
    temp1 = np.sqrt(temp)
    return temp1

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)



def findindex(org, x):
    result = []
    for k,v in enumerate(org): 
        if v == x:
            result.append(k)
    return result




def selNS(pop,k,ee):###################################store the solution who have the same classifiction error
    if len(pop) == k:
        return pop
    pop_fit = np.array([ind.fitness.value for ind in pop])
    index = np.argsort(pop_fit)################sortings' index
    fit_sort = sorted(pop_fit)##fitness' sorting
    # print('index',index)
    # print('fit_sort',fit_sort)
    if abs(fit_sort[k - 1] - fit_sort[k]) <= ee:####
         have_preserve_length = len(np.argwhere((fit_sort[k - 1]- fit_sort) > ee))
         # print('index[:have_preserve_length]',index[:have_preserve_length],have_preserve_length)
         off = [pop[m1] for m1 in index[:have_preserve_length]]
         need_more_length = k-have_preserve_length
         # print('need_more_length',need_more_length)
         index_fitness = np.argwhere(abs(fit_sort - fit_sort[k - 1]) <= ee)
         # print(index_fitness)
         list1 = []
         for ii in index_fitness:
             iii = random.choice(ii)
             list1.append(iii)
         list2 = [index[t] for t in list1]####from list2 choose need_more_length solutions to off
         # print(list2)
         # exit()
         pop_list2 = [pop[m2] for m2 in list2]
         size_solutions_in_pop_list2 = obtain_size(pop_list2)
         index1 = np.argsort(size_solutions_in_pop_list2)  ################sortings' index
         need_index = index1[:need_more_length]
         need_save = [list2[m3] for m3 in need_index]
         [off.append(pop[m4]) for m4 in need_save]
    else:
        off = [pop[k] for k in index[:k]]
    return off


def obtain_size(pop):
    size = []
    for z in pop:
       tt_01 = 1 * (np.array(z) >= 0.6)
       tt_01 = "".join(map(str, tt_01))  ######## the '0101001' of the current individual
       z_index = np.array(list(find_all(tt_01, '1')))  ##### the selected features of the current individual
       size.append(len(z_index))
    return size


def selection_compared_with_nearest(old,new):
    pop = toolbox.clone(new)
    dis1 = np.zeros((len(new),len(old)))
    for i in range(len(new)):
        for j in range(len(old)):
            dis1[i, j] = euclidean_distance(new[i], old[j])
        min_index = np.argwhere(dis1[i, :] == min(dis1[i, :]))
        min_one = random.choice(min_index)
        min_one = random.choice(min_one)
        temp = new[i].fitness.value-old[min_one].fitness.value
        if temp > 0:
            pop[i] = old[min_one]
        else:
            pop[i] = new[i]
    return pop


def selection_with_current(old,new,e):
    pop = toolbox.clone(new)
    for i in range(len(new)):
        temp = new[i].fitness.value - old[i].fitness.value
        if temp> 0:
            pop[i] = old[i]
        else:
            pop[i] = new[i]
    return pop





def mutDE(y, a, b, c, f):###mutation:DE/rand/1; if a is the best one, it will change as DE/best/1
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i]-c[i])
    return y


def mutDE1(y, a, b, c, d, e,f):###DE/rand-to-best/1;    DE/rand/2;  DE/best/2
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i]-c[i]) + f*(d[i]-e[i])
    return y


def cxBinomial(x, y, cr):#####binary crossover
    y_new = toolbox.clone(y)
    size = len(x)
    index = random.randrange(size)
    for i in range(size):
        if i == index or random.uniform(0, 1) <= cr:
            y_new[i] = y[i]
            #y_new[i] = 0.8*(1- y[i])
        else:
            y_new[i] = x[i]
    return y_new


def cxExponential(x, y, cr):####Exponential crossover
    size = len(x)
    index = random.randrange(size)
    # Loop on the indices index -> end, then on 0 -> index
    for i in chain(range(index, size), range(0, index)):
        x[i] = 0.8*(1-y[i])
        if random.random() < cr:
            break
    return x


##niche_index stores the index of individuals, the first one is offspring[ii]
def mutDE_LBP_NGI(offspring,niche_index,ii,f_mu):
    niche_offspring = [offspring[t] for t in niche_index]
    member = offspring[ii]
    niche_offspring_fit = [ind.fitness.value for ind in niche_offspring]
    min_index = np.argwhere(niche_offspring_fit == min(niche_offspring_fit))
    the_index_minimal_in_niche = niche_index[min_index[0][0]]
    y_new = toolbox.clone(member)
    if the_index_minimal_in_niche == ii:####the current is minal one
        #print('case1')
        nbest = member
        niche_offspring1 = [offspring[t] for t in niche_index[1:]]
        r1, r2 = random.sample(niche_offspring1, 2)  ####two uniuqe individuals
        for i2 in range(len(y_new)):
            y_new[i2] = member[i2] + f_mu * (r1[i2] - r2[i2])
    else:
        #print('case2')
        nbest = offspring[the_index_minimal_in_niche]
        offspring1 = toolbox.clone(offspring)
        if member == nbest:
            offspring1.remove(member)
        else:
            offspring1.remove(member)
            offspring1.remove(nbest)
        in1, in2 = random.sample(offspring1, 2)
        for i2 in range(len(y_new)):
           y_new[i2]=member[i2] +f_mu*(nbest[i2]-member[i2])+f_mu*(in1[i2]-in2[i2])
    return y_new,nbest
