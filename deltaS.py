from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# Load the normalized tree density data
data_path = 'dataset/normalized_tree_density.npy'
density_data = np.load(data_path)

# Reshape the data for GaussianMixture model
density_data = density_data.reshape(-1, 1)

# Fit a Gaussian Mixture Model with two components to the density data
gmm_density = GaussianMixture(n_components=2, random_state=0)
gmm_density.fit(density_data)

# Assuming norm_size is the mean of the density_data for demonstration purposes
norm_size = np.mean(density_data)
var_size = np.var(density_data)

# Generate samples from the 'size' distribution
size_samples = np.random.normal(norm_size, np.sqrt(var_size), 10000)

# Generate samples from the 'density' distribution using the GMM
density_samples = gmm_density.sample(10000)[0].flatten()

# Generate samples for 'X' based on the conditional distributions
x_given_size_samples = np.random.normal(density_samples, density_samples / 4)
x_given_density_samples = np.random.normal(size_samples, size_samples / 4)

# The final samples for 'X' are a mixture of 'x_given_size_samples' and 'x_given_density_samples'
x_samples = (x_given_size_samples + x_given_density_samples) / 2

# Estimate the PDF of 'X' using the samples
plt.figure(figsize=(10, 6))
plt.hist(x_samples, bins=100, density=True, alpha=0.6, color='g')
plt.title('Estimated PDF of X using Monte Carlo Simulation')
plt.xlabel('X values')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# Calculate the KDE for 'X' samples
x_kde = gaussian_kde(x_samples)

# Plot the KDE
x_values = np.linspace(min(x_samples), max(x_samples), 1000)
pdf_x = x_kde(x_values)
path = 'resultset/deltaS.jpg'
plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_x, label='KDE of X')
plt.title('KDE Estimated PDF of X')
plt.xlabel('X values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig(path)
plt.show()
