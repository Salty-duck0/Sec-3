import numpy as np

def generate_feedback_matrix(N, L, feedback_density, feedback_range):
    Wfb = np.zeros((N, L))  # Adjust the dimensions

    # Populate Wfb matrix with random values based on feedback_density
    for i in range(N):
        for j in range(L):
            p = np.random.random()
            if p < feedback_density:
                Wfb[i, j] = np.random.uniform(-feedback_range, feedback_range)

    return Wfb
