from PIL import Image
import numpy as np
from scipy.ndimage import generic_filter

# Function to normalize the pixel data
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Define the function to be applied to each pixel
def filter_func(values):
    # Check if the central pixel (13th in the 5x5 grid) is white (value of 0 in our case)
    if values[12] == 0:
        # Exclude the central pixel value when calculating the mean
        surrounding = np.delete(values, 12)
        # Calculate the mean of the surrounding pixel values
        mean_val = surrounding[surrounding > 0].mean() if surrounding[surrounding > 0].size > 0 else 0
        return mean_val
    else:
        # If the pixel is not white, return its original value
        return values[12]

# Open the image and convert it to a numpy array
image_path = 'map.jpg'
with Image.open(image_path) as img:
    img_array = np.array(img)

# Assuming the green channel represents tree density,
# extract the green channel
green_channel = img_array[:, :, 1]

# Apply the filter to the green channel of the image
# Using 'mirror' mode to handle the borders of the image
filtered_green_channel = generic_filter(green_channel, filter_func, size=(5,5), mode='mirror')

# Normalize the filtered tree density values to a range of [0, 1]
normalized_filtered_tree_density = normalize_data(filtered_green_channel)

# Output a small part of the matrix for a quick look
normalized_filtered_tree_density[:5, :5]
output_path = 'normalized_tree_density.npy'

np.save(output_path, normalized_filtered_tree_density)

