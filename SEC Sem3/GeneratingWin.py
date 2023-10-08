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

