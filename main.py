import map
from tree import Tree
import numpy as np
import evaluate

#初始化地图，对于亚马逊森林
density_array = np.load('dataset/normalized_tree_density.npy')
forest = [[Tree(0,0,0,0) for _ in range(density_array.shape[0])] for _ in range(density_array.shape[1])]
alpha = 1.5
norm_size = 10
norm_age = 300
map.init_map(density_array,forest,norm_size,norm_age,alpha)