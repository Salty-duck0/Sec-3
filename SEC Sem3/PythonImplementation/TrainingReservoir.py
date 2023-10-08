import numpy as np
from GeneratingW import generate_reservoir_matrix
from GeneratingWin import generate_input_matrix
from GeneratingWfb import generate_feedback_matrix

# Parameters
N = 10  # Number of nodes in the reservoir
K = 5   # Dimensionality of the input layer
L = 3   # Dimensionality of the output layer
T = 100  # Number of update steps
washout = 10  # Washout period, < T

density = 0.2  # Density of the random graph
spectral_radius = 1.0  # Desired spectral radius
binary_weights = False  # Set to True for binary connection weights
input_density = 0.3  # Density for input connections
input_range = 1.0    # Range for input weights
feedback_density = 0.4  # Density for feedback connections
feedback_range = 1.0    # Range for feedback weights


v = 0   #let
e = 0   #let
a = 0.1  #let

# Initialize matrices
W = generate_reservoir_matrix(N, density, spectral_radius, binary_weights=False)    # Reservoir weight matrix
Win = generate_input_matrix(N, K, input_density, input_range)    # Input weight matrix
Wfb = generate_feedback_matrix(N, L, feedback_density, feedback_range)# Feedback weight matrix

M = np.zeros((T - washout, N))              # State collecting matrix
Te = np.zeros((T - washout, L))              # Teacher collecting matrix
x = np.zeros(N)                             # Initial state vector


# Generate training input and target data (u(t) and y(t))
u = np.random.rand(T, K)  # Replace with your actual input data
y = np.random.rand(T, L)  # Replace with your actual target data


# Drive the reservoir and collect state vectors
for t in range(T):
    # Compute the new state vector using reservoir dynamics 
    U = Win @ u[t] + W @ x + Wfb @ y[t] + v + e
    x = (1 - a) * x + a * np.tanh(U)
 
    # Collect state vectors after the washout period
    if t >= washout:
        M[t - washout] = x

    # Collect training output data after the washout period
    if t >= washout:
        Te[t - washout] = y[t]

# Now you have M (state collecting matrix) and T (teacher collecting matrix)
print("State Collecting Matrix (M):")
print(M)
print("Teacher Collecting Matrix (T):")
print(Te)



def compute_wout(M, T, ridge_alpha=0.0):
    # Ridge Regression
    I = np.eye(M.shape[1])  # Identity matrix of appropriate size
    return np.linalg.inv(M.T @ M + ridge_alpha * I) @ M.T @ T

Wout = compute_wout(M , Te)
print("Wout: ")
print(Wout)















