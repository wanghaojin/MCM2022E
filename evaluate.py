import math
from tree import Tree
import deltaScal as ds
import joblib
from tqdm import tqdm
import copy

gmm_model_path = 'dataset/gmm_density_model.pkl'
gmm_density = joblib.load(gmm_model_path)

def deltas(map,x,y,width,height):
    sum = 0
    squares = 0
    if y > 0:
        squares += 1
        sum += map[x][y].density
    if x < height:
        squares += 1
        sum == map[x+1][y].density
    if x > 0:
        squares += 1
        sum += map[x-1][y].density
    if y < width:
        squares += 1
        sum += map[x][y+1].density
    average = sum / squares
    delta = ds.calculate_deltaS(map[x][y].size,average,gmm_density)
    return delta
def new_density(map,x,y,delta):
    current = map[x][y].density
    new = (1-current)*(delta ** 10) + current
    return new
def change_age(map,x,y):
    age = map[x][y].age
    return age+1
def simulation(map,duration,width,height):
    map_copy = copy.deepcopy(map)
    for year in tqdm(range(duration)):
        for x in range(height):
            for y in range(width):
                delta = deltas(map,x,y,width,height)
                if delta < 0:
                    delta = 0
                map_copy[x][y].size = map[x][y].size * (1 + delta**15)
                map_copy[x][y].density = new_density(map,x,y,delta)
                map_copy[x][y].age = change_age(map,x,y)
        map = map_copy
    return map