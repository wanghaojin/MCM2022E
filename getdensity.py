import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter

def is_white(value, threshold=180):
    return value > threshold

def apply_blur_filter(image_array, filter_size=9):
    def filter_func(values,threshold = 180):
        # 如果中心像素是白色，使用非白色的周围像素的均值
        if is_white(values[len(values) // 2]):
            non_white_values = values[values < threshold]
            return np.mean(non_white_values) if non_white_values.size > 0 else values[len(values) // 2]
        else:
            return values[len(values) // 2]
    
    return generic_filter(image_array, filter_func, size=(filter_size, filter_size))

# 读取图像并转换为灰度数组
with Image.open('map.jpg') as img:
    img_array = np.array(img.convert('L'))

# 应用模糊过滤器，尝试不同的filter_size和迭代次数以充分模糊文字
filter_size = 25  # 增加过滤器尺寸以覆盖更大的文字
iterations = 200   # 可能不需要200次迭代

for _ in range(iterations):
    img_array = apply_blur_filter(img_array, filter_size=filter_size)

# 归一化处理后的图像数组
img_array_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())

# 保存归一化的numpy矩阵
npy_output_path = 'dataset/normalized_tree_density.npy'
np.save(npy_output_path, img_array_normalized)

# 将处理后的图像数组转换回图像
output_image = Image.fromarray((img_array_normalized * 255).astype(np.uint8))

# 保存处理后的图像
output_image_path = 'resultset/check.png'
output_image.save(output_image_path)

