import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array from the file
npy_path = 'dataset/normalized_tree_density.npy'
array = np.load(npy_path)

# Create a histogram of the array values
plt.figure(figsize=(10, 6))
plt.hist(array.ravel(), bins=50, color='blue', alpha=0.7)  # Use ravel to flatten the array
plt.title('Histogram of Tree Density Values')
plt.xlabel('Normalized Density Value')
plt.ylabel('Frequency')

# Save the histogram
histogram_path = 'resultset/histogram.png'
plt.savefig(histogram_path)

