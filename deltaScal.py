import joblib
from scipy.stats import norm

gmm_model_path = 'dataset/gmm_density_model.pkl'
gmm_density = joblib.load(gmm_model_path)

new_size = 0.5  
new_density = 0.7  
def relu(x):
    if x < 0:
        return 0
    else:
        return x
def calculate_deltaS(size, density, gmm_model):
    size = relu(size)
    density = relu(density)
    weights = gmm_model.weights_
    means = gmm_model.means_.flatten()
    covariances = gmm_model.covariances_.flatten()
    if size < 0:
        print("size:",size)
    x_given_size = relu(norm(loc=density, scale=density/4).rvs())
    x_given_density = relu(norm(loc=size, scale=size/4).rvs())

    deltaS = (x_given_size + x_given_density) / 2
    return deltaS

# deltaS = calculate_deltaS(new_size, new_density, gmm_density)
# print(deltaS)
