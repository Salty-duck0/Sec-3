import numpy as np

# Define the parameters
N = 3  # Number of nodes in the reservoir
K = 1  # Dimensionality of the input layer
L = 1  # Dimensionality of the output layer
T = 300  # Total number of time steps


# Provided reservoir weight matrix (W) and input weight matrix (Win)
W = np.array([[2.0, -0.86, 0.0],
              [-0.72, 0.0, -0.4],
              [1.04, 0.0, 0.0]])

Win = np.array([[2.0],
                [0.4],
                [-0.49]])



# Initialize state vector
x = np.zeros(N)

# Create u(t) and y(t) based on the provided functions
t = np.arange(0, 3, 0.01)
u = 0.5 * np.sin(8 * np.pi * t).reshape(-1, 1)
y = np.sin(8 * np.pi * t).reshape(-1, 1)


np.savetxt('u_dataNewEx.txt', u)
np.savetxt('y_dataNewEx.txt', y)


M = np.zeros((T,N))    # State Collecting matrix
Te = np.zeros((T , L))  # teacher collecting matrix

print(M.shape)
print(Te.shape)


# Train the toy model with the specified washout period
for t in range(T):
    
    # Compute the new state vector using reservoir dynamics
    U = Win @ u[t] + W @ x
    x = np.tanh(U)
    

    # Collect the state vectors after the washout period

    M[t ] = x   # Store state vectors correctly in the list
    Te[t] = y[t]


print(M)
print(Te)

def compute_wout(M, T, ridge_alpha=0.0):
    # Ridge Regression
    I = np.eye(M.shape[1])  # Identity matrix of appropriate size
    return np.linalg.inv(M.T @ M + ridge_alpha * I) @ M.T @ T

Wout = compute_wout(M , Te)
print("Wout: ")
print(Wout)




print(W.shape)
print(Win.shape)
print(M.shape)
print(Te.shape)
print(Wout.shape)
print(x.shape)






#Run_RC  to predict y for new values of x


def run_RC(W, Win, Wout, N, K, L, input,M, time,a =0.05):
    if input is None:
        input = np.zeros((time * K,))
    
    x = M[-1, :]      #(3x1)
    y = np.zeros((L,))    #(1x1)
    states = np.zeros((time, N))     #(300 x3)
    outputs = np.zeros((time, L))    #(300,1)

    for t in range(time):
        
        U = Win @ input[t] + W @ x    #(3,1)
        x = ((1 - a) * x + a * np.tanh(U) )  #(3,1)
        print(x.shape)
        print(Wout)
        print(x)
        y = np.dot(Wout.T, x)     #(Scalar  since dot product)
        states[t, :] = x
        outputs[t, :] = y      #(300,1)

    return outputs

y_pred = M @ Wout
print(y_pred)

np.savetxt('y_predNewEx.txt',y_pred)



ti = np.arange(3, 6, 0.01)
input = 0.5 * np.sin(8 * np.pi * ti).reshape(-1, 1)


print("running params")
print(input.shape)
print(x.shape)
print(Wout.shape)


new_out = run_RC(W, Win,Wout,N,K,L,input,M,300,0.05)

np.savetxt('x_dataNewExNewInput.txt', input)
np.savetxt('y_dataNewExNewOutput.txt', new_out)