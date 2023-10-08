import numpy as np
import networkx as nx

def generate_reservoir_matrix(N, density, spectral_radius, binary_weights=False):
    # Create a random graph with specified density using NetworkX
    G = nx.gnp_random_graph(N, density)

    # Convert the graph to an adjacency matrix
    W0 = nx.adjacency_matrix(G).toarray()

    # Create a random matrix for rescaling
    random_matrix = np.random.uniform(-1, 1, size=(N, N))

    if binary_weights:
        # Create a random matrix of probabilities
        p_matrix = np.random.random(size=(N, N))

        # Set positive or negative values based on probabilities
        W0 = np.where(p_matrix < 0.5, -1, 1) * W0
    else:
        # Multiply by a random number from the uniform distribution
        W0 = W0 * random_matrix

    # Calculate the maximum eigenvalue of W0
    max_eigenvalue = max(np.linalg.eigvals(W0), key=abs)

    # Rescale the matrix to achieve the desired spectral radius
    W = (spectral_radius / abs(max_eigenvalue)) * W0

    return W

# Example usage:
N = 10  # Number of nodes in the reservoir
density = 0.2  # Density of the random graph
spectral_radius = 1.0  # Desired spectral radius
binary_weights = False  # Set to True for binary connection weights

W = generate_reservoir_matrix(N, density, spectral_radius, binary_weights)
print(W)
