import map
from tree import Tree
import numpy as np
import evaluate
from PIL import Image

def is_white(value, threshold=180):
    return value > threshold

def apply_blur_filter(image_array, filter_size=9):
    def filter_func(values, threshold=180):
        if is_white(values[len(values) // 2]):
            non_white_values = values[values < threshold]
            return np.mean(non_white_values) if non_white_values.size > 0 else values[len(values) // 2]
        else:
            return values[len(values) // 2]
    return generic_filter(image_array, filter_func, size=(filter_size, filter_size))

def reconstruct_color_image(normalized_gray_array, original_color_array, density_normalized):
    reconstructed_color_array = np.zeros(original_color_array.shape, dtype=np.uint8)
    for i in range(normalized_gray_array.shape[0]):
        for j in range(normalized_gray_array.shape[1]):
            gray_value = normalized_gray_array[i, j]
            brightness_factor = density_normalized[i, j] / gray_value if gray_value != 0 else 0
            reconstructed_color_array[i, j] = np.clip(original_color_array[i, j] * brightness_factor, 0, 255)
    return reconstructed_color_array

# 初始化地图，对于亚马逊森林
density_array = np.load('dataset/normalized_tree_density.npy')
width = density_array.shape[1]  # 720
height = density_array.shape[0]  # 600
forest = [[None for _ in range(width)] for _ in range(height)]
alpha = 1.5
norm_size = 10
norm_age = 300
map.init_map(density_array, forest, norm_size, norm_age,alpha)

# 模拟森林生长
year = 100
new_forest = evaluate.simulation(forest, year, width, height)

# 重建彩色图像
new_density_array = np.array([[tree.density for tree in row] for row in new_forest])
new_density_normalized = (new_density_array - np.min(new_density_array)) / (np.max(new_density_array) - np.min(new_density_array))
with Image.open('dataset/map.jpg') as img:
    original_color_image = img
    original_color_array = np.array(original_color_image)
gray_array = np.mean(original_color_array, axis=2)
gray_array_normalized = (gray_array - np.min(gray_array)) / (np.max(gray_array) - np.min(gray_array))
reconstructed_color_array = reconstruct_color_image(gray_array_normalized, original_color_array, new_density_normalized)
new_color_image = Image.fromarray(reconstructed_color_array)
new_color_image_path = 'resultset/new_forest_color_image.png'
new_color_image.save(new_color_image_path)
new_color_image.show()
