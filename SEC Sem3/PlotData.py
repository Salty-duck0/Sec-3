import matplotlib.pyplot as plt
import numpy as np

# Initialize empty lists to store the data
x = np.linspace(0, 100, 300)
y = []
y_pred = []


# Read y values from the text file and ignore the first 50 data points
with open('y_data.txt', 'r') as file:
    for i, line in enumerate(file):
            y.append(float(line.strip()))

# Read y_pred values from the text file
with open('y_pred.txt', 'r') as file:
    for line in file:
        y_pred.append(float(line.strip()))

# Create a new figure
plt.figure()

# Plot the actual values (y) in blue
plt.plot(x, y, label='Actual', color='blue')

# Plot the predicted values (y_pred) in red
plt.plot(x, y_pred, label='Predicted', color='red')

# Add labels and a legend
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.legend()

# Show the plot
plt.show()
