import numpy as np
from tree import Tree
import random

def init_map(density_array,forest,norm_size,norm_age,alpha):
    for x in range(density_array.shape[0]):
        for y in range(density_array.shape[1]):
            size = random.normalvariate(norm_size,norm_size/4)
            age = random.normalvariate(norm_age,norm_age/4)
            forest[x][y] = Tree(size,density_array[x][y],age,alpha)