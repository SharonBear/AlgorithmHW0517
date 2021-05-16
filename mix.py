# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:10:39 2021

@author: sharon
"""

import numpy as np
import sys
import random
from datetime import datetime
from sys import maxsize
from itertools import permutations
from decimal import Decimal
import matplotlib.pyplot as plt

node = np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

# implementation of traveling Salesman Problem
def DP(graph):
 
    # store all vertex apart from source vertex
    vertex = []
    for i in range(n):
        if i != 0:
            vertex.append(i)
 
    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation=permutations(vertex)
    for i in next_permutation:
 
        # store current Path weight(cost)
        current_pathweight = 0
 
        # compute current path weight
        k = 0
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][0]
 
        # update minimum
        min_path = min(min_path, current_pathweight)
         
    return min_path
 
    

    
# %%
class TSPProblem:
    def __init__(self, coordinate, cities_name):
        self.coordinate = coordinate
        self.cities_name = cities_name

    def get_distance(self, arr1, arr2):
        # Euclidean distance
        #return np.sqrt(np.power(arr1 - arr2, 2).sum())
        # definded distance
        return self.coordinate[arr1][arr2]

    def compute_objective_value(self, cities_id):
        total_distance = 0
        for i in range(len(cities_id)):
            city1 = cities_id[i]
            city2 = cities_id[i + 1] if i < len(cities_id) - 1 else cities_id[0]
            #total_distance += self.get_distance(self.coordinate[city1], self.coordinate[city2])
            total_distance += self.get_distance(city1, city2)
        return total_distance

    def to_cities_name(self, cities_id):
        return [self.cities_name[i] for i in cities_id]


# %%
class AntSystem:
    def __init__(self, pop_size, coordinate, pheromone_drop_amount, evaporate_rate,
                 pheromone_factor, heuristic_factor,
                 get_distance, compute_objective_value):

        self.num_ants = pop_size
        self.coordinate = coordinate
        self.num_cities = len(coordinate)
        self.get_distance = get_distance
        self.compute_objective_value = compute_objective_value
        self.pheromone_drop_amount = pheromone_drop_amount
        self.evaporate_rate = evaporate_rate
        self.pheromone_factor = pheromone_factor
        self.visibility_factor = heuristic_factor

    def initialize(self):
        self.one_solution = np.arange(self.num_cities, dtype=int)
        self.solutions = np.zeros((self.num_ants, self.num_cities), dtype=int)
        for i in range(self.num_ants):
            for c in range(self.num_cities):
                self.solutions[i][c] = c

        self.objective_value = np.zeros(self.num_ants)
        self.best_solution = np.zeros(self.num_cities, dtype=int)
        self.best_objective_value = sys.float_info.max

        self.visibility = np.zeros((self.num_cities, self.num_cities))
        self.pheromone_map = np.ones((self.num_cities, self.num_cities))

        # heuristic_values
        for from_ in range(self.num_cities):
            for to in range(self.num_cities):
                if (from_ == to): continue
                #distance = self.get_distance(self.coordinate[from_], self.coordinate[to])
                distance = self.get_distance(from_, to)
                self.visibility[from_][to] = 1 / distance

    def do_roulette_wheel_selection(self, fitness_list):
        kk=0
        for i in range(len(fitness_list)-1):
            kk += fitness_list[i]
        transition_probability = [fitness / kk for fitness in fitness_list]

        rand = random.random()
        sum_prob = 0
        for i, prob in enumerate(transition_probability):
            sum_prob += prob
            if (sum_prob >= rand):
                return i

    def update_pheromone(self):
        # evaporate hormones all the path
        self.pheromone_map *= (1 - self.evaporate_rate)

        # Add hormones to the path of the ants
        for solution in self.solutions:
            for j in range(self.num_cities):
                city1 = solution[j]
                city2 = solution[j + 1] if j < self.num_cities - 1 else solution[0]
                self.pheromone_map[city1, city2] += self.pheromone_drop_amount

    def _an_ant_construct_its_solution(self):
        candidates = [i for i in range(self.num_cities)]
        # random choose city as first city
        current_city_id = random.choice(candidates)
        self.one_solution[0] = current_city_id
        candidates.remove(current_city_id)

        # select best from candiate
        for t in range(1, self.num_cities - 1):
            # best
            fitness_list = []
            for city_id in candidates:
                fitness = pow(self.pheromone_map[current_city_id][city_id], self.pheromone_factor) * \
                          pow(self.visibility[current_city_id][city_id], self.visibility_factor)
                fitness_list.append(fitness)

            next_city_id = candidates[self.do_roulette_wheel_selection(fitness_list)]
            candidates.remove(next_city_id)
            self.one_solution[t] = next_city_id

            current_city_id = next_city_id
        self.one_solution[-1] = candidates.pop()

    def each_ant_construct_its_solution(self):
        for i in range(self.num_ants):
            self._an_ant_construct_its_solution()
            for c in range(self.num_cities):
                self.solutions[i][c] = self.one_solution[c]

            self.objective_value[i] = self.compute_objective_value(self.solutions[i])

    def update_best_solution(self):
        for i, val in enumerate(self.objective_value):
            if (val < self.best_objective_value):
                for n in range(self.num_cities):
                    self.best_solution[n] = self.solutions[i][n]

                self.best_objective_value = val




# Driver Code
if __name__ == "__main__":
    file1 = open("output.txt","w")
    dpsum=np.ones((17, 2))
    assum=np.ones((17, 2))
    dpw=np.ones((17, 2))
    asw=np.ones((17, 2))
    differw=np.ones(17)
    dpavg=np.ones(17)
    asavg=np.ones(17)
    # \n is placed to indicate EOL (End of Line)
    for n in range(4,21):
        print(n)
        cal = 0
        cal2 = 0
        dpw[n-4][0]= n
        asw[n-4][0] = n
        print("------------  %d x %d  ------------"%(n,n),file=file1)
        i=1
        while i!=6:
            print("第 %d 次"%i,file=file1)
            print("第 %d 次"%i)
            a=np.random.randint(1,30,size=[n,n])
            for k in range(n):
                a[k][k] = 0
                h = k+1
                while k<n-1 and h!=n:
                    a[k][h] = a[h][k]
                    h+=1
            print(a,file=file1)
            
            d1=datetime.now().timestamp()
            w=DP(a)
            print("DP Weight:",w,file=file1)
            d2=datetime.now().timestamp()
            print("DP計算時間:",d2-d1,file=file1)
            cal +=Decimal(d2-d1)
            dpw[n-4][1]+=w
            

            problem = TSPProblem(a, node)
            
            pop_size = 20
            pheromone_drop_amount = 0.001
            evaporate_rate = 0.1
            pheromone_factor = 1
            heuristic_factor = 3
            
            solver = AntSystem(pop_size, a, pheromone_drop_amount, evaporate_rate,
                               pheromone_factor, heuristic_factor,
                               problem.get_distance, problem.compute_objective_value)
            
            solver.initialize()
            start = datetime.now().timestamp()
            solver.each_ant_construct_its_solution()
            solver.update_pheromone()
            solver.update_best_solution()
            w=int(solver.best_objective_value)
            print("AS Weight:",w,file=file1)
            end = datetime.now().timestamp()
            print("AS計算時間:",end-start,file=file1)
            cal2 +=Decimal(end-start)
            asw[n-4][1]+=w
            print("",file=file1)
            i+=1
            
        dpsum[n-4][0]=n
        dpsum[n-4][1]=cal/5
        dpavg[n-4]=cal/5
        assum[n-4][0]=n
        assum[n-4][1]=cal2/5
        asavg[n-4]=cal2/5
        print("",file=file1)
    diff=np.ones((17, 2))
    for i in range(17):
        diff[i][0]=dpw[i][0]
        diff[i][1]=(asw[i][1]-dpw[i][1])/(dpw[i][1])
        differw[i]= (asw[i][1]-dpw[i][1])/(dpw[i][1])
    print("DP平均時間:",file=file1)
    print(dpsum,file=file1)
    print("AS平均時間:",file=file1)
    print(assum,file=file1)
    print("Weight差:",file=file1)
    print(diff,file=file1)

    node1=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    plt.figure(figsize=(15, 10))
    plt.title("Difference of Weight") # y label
    plt.ylabel("Difference of Average Weight") # y label
    plt.xlabel("Number of Node") # x label
    plt.plot(node1,differw)
    plt.show()
    

    plt.figure(figsize=(15, 10))
    plt.title("DP average time") # y label
    plt.ylabel("Average Time") # y label
    plt.xlabel("Number of node") # x label
    plt.plot(node1,dpavg)
    plt.show()
    
    

    plt.figure(figsize=(15, 10))
    plt.title("AS average time") # y label
    plt.ylabel("Average Time") # y label
    plt.xlabel("Number of node") # x label
    plt.plot(node1,asavg)
    plt.show()
    
    
    file1.close()
