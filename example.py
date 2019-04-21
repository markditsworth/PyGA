#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 00:08:08 2018

@author: markditsworth
"""
import numpy as np

from PyGA import genetic

def fcn_(x,y):
    return 4597 - np.square(x-9) - np.square(y-22)

def toSixBitBinary(n):
    array = np.zeros(6)
    m = [2**x for x in range(6)]
    m.reverse()
    for i,p in enumerate(m):
        if n >= p:
            array[i] = 1
            n -= p
    return array

def fromSixBitBinary(array):
    m = [2**x for x in range(6)]
    m.reverse()
    return np.sum(np.multiply(m,array))

def encode(x,y):
    a = toSixBitBinary(x)
    b = toSixBitBinary(y)
    return np.hstack((a,b))

def decode(array):
    x = fromSixBitBinary(array[:6])
    y = fromSixBitBinary(array[6:])
    return x,y

def fcn(array):
    x,y = decode(array)
    return fcn_(x,y)/1000

population_size=25
p = np.ones((25,2))

population = np.empty((population_size,12))
for i in range(population_size):
    population[i,:] = encode(p[i,0],p[i,1])

pop = genetic(population,fcn,parent_probability=0.3,mutation_probability=0.001,iter_number=100,
              logging=True,report=True,selection_type='tournament')

solution, fitness = pop.evolve()

xx,yy = decode(solution)

print xx, yy
print fitness

        
    
    
    
    