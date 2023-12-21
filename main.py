import map
from tree import Tree
import random
import numpy as np

density_array = np.load('normalized_tree_density.npy')
forest = [[Tree(0,0,0) for _ in range(density_array.shape[0])] for _ in range(density_array.shape[1])]
alpha = 1.5
norm_size = 10
norm_age = 300
map.init_map(density_array,forest,norm_size,norm_age,alpha)