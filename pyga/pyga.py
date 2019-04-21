#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 18:56:02 2018

@author: markditsworth
"""
import numpy as np

class genetic:
    def __init__(self,population, objective_function, alphabet=None, parent_probability=0.5 ,
                 mutation_probability=0.01, iter_number=100, crossover_points=1,
                 selection_type='roulette', logging=False, report=False):
        assert selection_type in ['roulette','tournament'], 'invalid selection_type. roulette or tournament'
        
        self.P = population
        self.objFunc = objective_function
        self.alphabet = alphabet
        if alphabet == None:
            self.alphabet = set(population.flatten())
        self.p_c = parent_probability
        self.p_m = mutation_probability
        self.stopping = iter_number
        self.crossover_points = crossover_points
        self.selectionType = selection_type
        self.logging = logging
        self.report = report
    
    def feval(self):
        evals = np.zeros(self.P.shape[0])
        for i in range(self.P.shape[0]):
            evals[i] = self.objFunc(self.P[i,:])
        return evals
        
    def mutate(self,chromosome):
        for i,x in enumerate(chromosome):
            if np.random.random() < self.p_m:
                alpha = self.alphabet.copy()
                alpha.discard(x)
                random_value_idx = np.random.randint(0,len(alpha))
                chromosome[i] = list(alpha)[random_value_idx]
        return chromosome
    
    def crossoverSections(self,length,point_number):
        # initialize array
        crossover_pts = np.zeros(point_number,dtype=int)
        #limit of random integer generation
        stop = length - 1
        
        # generate array of crossover points
        for i,_ in enumerate(crossover_pts):
            cp = np.random.randint(0,stop)
            # make sure each crossover point is unique
            while cp in crossover_pts:
                cp = np.random.randint(0,stop)
            crossover_pts[i] = cp
        # sort from low to high
        crossover_pts.sort()
        
        # initilize list of tuples
        crossover_pts_tups = []
        i = 0
        point1 = 0
        while i<len(crossover_pts):
            point2 = crossover_pts[i]
            crossover_pts_tups.append((point1,point2))
            point1 = point2
            i+=1
        crossover_pts_tups.append((point1,length))
        
        return crossover_pts_tups
    
    def crossover(self,parent1,parent2,sections):
        parent1_chunks = sections[::2]
        parent2_chunks = sections[1::2]
        
        offspring1 = np.empty(len(parent1),dtype=parent1.dtype)
        
        for chunk in parent1_chunks:
            offspring1[chunk[0]:chunk[1]] = parent1[chunk[0]:chunk[1]]
        for chunk in parent2_chunks:
            offspring1[chunk[0]:chunk[1]] = parent2[chunk[0]:chunk[1]]
            
        offspring2 = np.empty(len(parent1),dtype=parent1.dtype)
        
        # reverse the parent chunk slices for second offspring
        for chunk in parent1_chunks:
            offspring2[chunk[0]:chunk[1]] = parent2[chunk[0]:chunk[1]]
        for chunk in parent2_chunks:
            offspring2[chunk[0]:chunk[1]] = parent1[chunk[0]:chunk[1]]
        
        return offspring1, offspring2
    
    def parentPairsIndex(self):
        N = self.P.shape[0]
        k = int(self.p_c * N / 2)
        return np.random.randint(0,N,size=(k,2))
        
    def roulette_selection(self,probabilities):
        M = np.empty(self.P.shape,dtype=self.P.dtype)
        
        #idx = 0
        for i in range(M.shape[0]):
            loop = True
            while loop:
                idx = np.random.randint(0,len(probabilities))
                if np.random.random() > probabilities[idx]:
                    M[i,:] = self.P[idx,:]
                    loop = False
        
        return M
    
    def tournament_selection(self,fitnesses):
        M = np.empty(self.P.shape,dtype=self.P.dtype)
        
        for i in range(M.shape[0]):
            a = np.random.randint(self.P.shape[0])
            b = np.random.randint(self.P.shape[0])
            if fitnesses[a] > fitnesses[b]:
                M[i,:] = self.P[a,:]
            else:
                M[i,:] = self.P[b,:]
                
        return M
    
    def evolve_(self,n):
        # evaluate
        evaluations = self.feval()
        if self.logging:
            self.log(evaluations,n)
        
        # selection
        if self.selectionType == 'roulette':
            Sum = np.sum(evaluations)
            if Sum == 0:
                probs = np.ones(len(evaluations))*0.1
            else:
                probs = evaluations / Sum
            M = self.roulette_selection(probs)
            
        else:
            M = self.tournament_selection(evaluations)
        
        # crossover
        parentPairings = self.parentPairsIndex()
        for pair in parentPairings:
            offspring1, offspring2 = self.crossover(M[pair[0],:], M[pair[1],:],
                                                   self.crossoverSections(self.P.shape[1],self.crossover_points))
            
            M[pair[0],:] = offspring1
            M[pair[1],:] = offspring2
        
        # mutate
        for i in range(M.shape[0]):
            M[i,:] = self.mutate(M[i,:])
        
        # update
        self.P = M
    
    def evolve(self):
        i = 0
        while i < self.stopping:
            self.evolve_(i)
            i+=1
        final_evals = self.feval()
        
        if self.logging:
            self.log(final_evals,i)
        
        best_eval = np.max(final_evals)
        idx = np.argmax(final_evals)
        best_chromosome = self.P[idx,:]
        
        if self.report:
            self.graph(self.stopping)
            
        return best_chromosome, best_eval
    
    def log(self,evals,iter_num):
        np.save('evaluations_%d.npy'%iter_num,evals)
    
    def graph(self,total_iterations):
        best = np.zeros(total_iterations)
        worst = np.zeros(total_iterations)
        average = np.zeros(total_iterations)
        
        for i in range(total_iterations):
            evals = np.load('evaluations_%d.npy'%i)
            best[i] = np.max(evals)
            worst[i] = np.min(evals)
            average[i] = np.mean(evals)
        
        import matplotlib.pyplot as plt
        x = np.arange(total_iterations)
        plt.scatter(x,best,c='b',label='best')
        plt.scatter(x,worst,c='r',label='worst')
        plt.plot(x,average,c='k',ls='--',label='average')
        plt.legend()
        plt.title('Fitness of population over time')
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.show()
    
    
    