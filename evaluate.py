import numpy as np
import joblib
from tqdm import tqdm
from tree import Tree
import deltaScal as ds

gmm_model_path = 'dataset/gmm_density_model.pkl'
gmm_density = joblib.load(gmm_model_path)

def logistic_function(x, L=1, k=10, x0=0.5):
    return L / (1 + np.exp(-k * (x - x0)))

def deltas(map, x, y, width, height):
    neighbors = []
    if y > 0:
        neighbors.append(map[x][y - 1].density)
    if x < height - 1:
        neighbors.append(map[x + 1][y].density)
    if x > 0:
        neighbors.append(map[x - 1][y].density)
    if y < width - 1:
        neighbors.append(map[x][y + 1].density)
    
    average = np.mean(neighbors) if neighbors else 0
    return ds.calculate_deltaS(map[x][y].size, average, gmm_density)


def simulation(map, duration, width, height):
    for year in tqdm(range(duration)):
        for x in range(height):
            for y in range(width):
                delta = deltas(map, x, y, width, height)
                delta = max(0, delta)  
                cals = logistic_function(delta)
                map[x][y].size *= (1 + cals ** 15)
                map[x][y].density = (1 - map[x][y].density) * (cals ** 10) + map[x][y].density
                map[x][y].age += 1
    return map
