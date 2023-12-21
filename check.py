import numpy as np
from PIL import Image

image_path = 'map.jpg'
with Image.open(image_path) as img:
    array1 = np.array(img)
max_val = array1.max() 
min_val = array1.min()  
array2 = np.load('normalized_tree_density.npy')
print(array2.min())
print(array2.max())
print(array2.shape)
check = (max_val - min_val) * array2 + min_val
check_uint8 = check.astype(np.uint8)
image = Image.fromarray(check_uint8)

path = 'check.png'
image.save(path)
