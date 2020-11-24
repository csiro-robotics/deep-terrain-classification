import functools
import numpy as np
import sys
import csv

def get_next_batch(x, y, l, start, end):
    x_batch = x[start:end,:,:]
    y_batch = y[start:end,:]
    l_batch = l[start:end]
    return x_batch, y_batch, l_batch

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def read_lines(file):
    with open(file, newline="") as data:
        reader = csv.reader(data)
        ind = 0
        for row in reader:
            if(ind > 0):
                yield [float(i) for i in row]                 
            ind+=1

def step_count(raw_inp, num_trials, num_steps):
    cnt = 0
    inputs = [[] for i in range(num_trials)]
    for i in range(raw_inp.shape[0]):
        if i > 0:
            if (raw_inp[i,3] != raw_inp[i-1,3]):# 3 is the column in csv files that shows the num of tiral     
                cnt += 1
        inputs[cnt].append(raw_inp[i])
    minimum = 1000000
    for i in range(num_trials):
        if (len(inputs[i]) < minimum):
            minimum = len(inputs[i])
    each_step = np.floor(minimum/num_steps)
    return each_step, inputs


