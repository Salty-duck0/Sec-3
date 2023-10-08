import numpy as np
from GeneratingW import generate_reservoir_matrix
from GeneratingWin import generate_input_matrix
from GeneratingWfb import generate_feedback_matrix

# TRAINING RESERVOIR

# Define the parameters
N = 3  # Number of nodes in the reservoir
K = 1  # Dimensionality of the input layer
L = 1  # Dimensionality of the output layer
T = 300  # Total number of time steps

density = 0.5  # Density of the random graph
spectral_radius = 1.0  # Desired spectral radius
binary_weights = False  # Set to True for binary connection weights
input_density = 0.5  # Density for input connections
input_range = 1.0    # Range for input weights
feedback_density = 0.8  # Density for feedback connections
feedback_range = 1.0    # Range for feedback weights



# Initialize matrices
W = generate_reservoir_matrix(N, density, spectral_radius, binary_weights=False)    # Reservoir weight matrix
Win = generate_input_matrix(N, K, input_density, input_range)    # Input weight matrix
Wfb = generate_feedback_matrix(N, L, feedback_density, feedback_range)# Feedback weight matrix
# W = np.array([[ 0.        , -0.        , -0.        ],
#               [-0.        , -0.        , -0.90252262],
#               [-0.        , -1.10800547, -0.        ]])

# Win = np.array([[ 0.        ],
#                 [-0.34690941],
#                 [-0.18151322]])

# Wfb = np.array([[-0.66766746],
#                 [ 0.03278151],
#                 [-0.62120842]])
print(W)
print(Win)
print(Wfb)


M = np.zeros((T, N))              # State collecting matrix
Te = np.zeros((T, L))              # Teacher collecting matrix
x = np.zeros(N)                    # Initial state vector


e = 1  # The constant bias term serves to push the activation function into the nonlinear or saturation regimes 
v = 0.09    #A small constant noise term ν, typically in the range of ν = 0.001 → 0.01, is often added to the “input” during output weight training.
a = 0.08    #We also have a, a “leak” term between 0 and 1. 

# Create u(t) and y(t) based on the provided functions
t = np.arange(0, 3, 0.01)
u = 0.5 * np.sin(8 * np.pi * t).reshape(-1, 1)
y = np.sin(8 * np.pi * t).reshape(-1, 1)

np.savetxt('u_dataNewEx.txt', u)
np.savetxt('y_dataNewEx.txt', y)


# Drive the reservoir and collect state vectors
for t in range(T):
    # Compute the new state vector using reservoir dynamics 
    U = Win @ u[t] + W @ x + Wfb @ y[t] + v 
    x = (1 - a) * x + a * np.tanh(U+e)
 
    # Collect state vectors 
    M[t] = x

    # Collect training output data 
    Te[t ] = y[t]

# Now you have M (state collecting matrix) and T (teacher collecting matrix)
print("State Collecting Matrix (M):")
print(M)
print("Teacher Collecting Matrix (T):")
print(Te)



def compute_wout(M, T, ridge_alpha=1.0):
    # Ridge Regression
    I = np.eye(M.shape[1])  # Identity matrix of appropriate size
    return np.linalg.inv(M.T @ M + ridge_alpha * I) @ M.T @ T

Wout = compute_wout(M , Te)
print("Wout: ")
print(Wout)




#RUNNING RESERVOIR   


def run_RC(W, Win, Wout, N, K, L, input,M, time,a = 0.5):
    if input is None:
        input = np.zeros((time * K,))
    
    x = M[-1, :]      #(3x1)
    y = np.zeros((L,))    #(1x1)
    states = np.zeros((time, N))     #(300 x3)
    outputs = np.zeros((time, L))    #(300,1)

    for t in range(time):
        
        U = Win @ u[t] + W @ x + Wfb @ y + v 
        x = (1 - a) * x + a * np.tanh(U+e)
        y = np.dot(Wout.T, x)     #(Scalar  since dot product)
        states[t, :] = x
        outputs[t, :] = y      #(300,1)

    return outputs



ti = np.arange(3, 6, 0.01)
input = 0.5 * np.sin(8 * np.pi * ti).reshape(-1, 1)

new_out = run_RC(W, Win,Wout,N,K,L,input,M,300,0.85)

np.savetxt('x_dataNewExNewInput.txt', input)
np.savetxt('y_dataNewExNewOutput.txt', new_out)









