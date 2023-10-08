import numpy as np

def generate_input_matrix(N, K, input_density, input_range):
    Win = np.zeros((N, K))  # Adjust the dimensions

    # Populate Win matrix with random values based on input_density
    for i in range(N):
        for j in range(K):
            p = np.random.random()
            if p < input_density:
                Win[i, j] = np.random.uniform(-input_range, input_range)

    return Win

# Example usage for Win:
N = 10  # Number of nodes in the hidden network layer
K = 5   # Dimensionality of the input layer

input_density = 0.3  # Density for input connections
input_range = 1.0    # Range for input weights

Win = generate_input_matrix(N, K, input_density, input_range)
print("Win:")
print(Win)
