from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import joblib

# Load the normalized tree density data
data_path = 'dataset/normalized_tree_density.npy'  # Use the correct path where the .npy file is located
data = np.load(data_path)

# Reshape the data for GaussianMixture model
data = data.reshape(-1, 1)

# Fit a Gaussian Mixture Model with two components
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data)

# Retrieve the model parameters
weights = gmm.weights_
means = gmm.means_.flatten()
covariances = gmm.covariances_.flatten()

# Define a range of x values for plotting the PDF
x_values = np.linspace(0, 1, 1000)  # Assuming the data is normalized between 0 and 1

# Calculate the PDF of x as a weighted sum of the two Gaussian components
pdf_x = np.zeros_like(x_values)
for weight, mean, covar in zip(weights, means, covariances):
    pdf_x += weight * norm.pdf(x_values, mean, np.sqrt(covar))

# Plot the PDF of x
plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_x, label='PDF of x based on GMM')
plt.title('Probability Density Function of x')
plt.xlabel('x values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = 'resultset/distribution.png'  # Use the correct path to save the plot
plt.savefig(plot_path)
plt.close()

# Save the GMM model
gmm_model_path = 'dataset/gmm_density_model.pkl'  # Use the correct path to save the model
joblib.dump(gmm, gmm_model_path)  # Save the 'gmm' object, not 'gmm_density'
