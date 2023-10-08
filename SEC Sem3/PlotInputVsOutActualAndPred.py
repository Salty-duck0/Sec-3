import matplotlib.pyplot as plt
import numpy as np

# Initialize empty lists to store the data
time = np.linspace(0, 100, 300)
x_old = []
y_old = []

print(x_old)

x_new = []
y_new = []



y_pred_new = []


# Read y values from the text file and ignore the first 50 data points
with open('u_dataNewEx.txt', 'r') as file:
    for i, line in enumerate(file):
        x_old.append(float(line.strip()))

# Read y_pred values from the text file
with open('y_dataNewEx.txt', 'r') as file:
    for line in file:
        y_old.append(float(line.strip()))
        
        
with open('x_dataNewExNewInput.txt', 'r') as file:
    for i, line in enumerate(file):
        x_new.append(float(line.strip()))

# Read y_pred values from the text file
with open('y_dataNewExNewOutput.txt', 'r') as file:
    for line in file:
        y_new.append(float(line.strip()))
        
        
# Read y_predNewValuesOldInput values from the text file
with open('y_predNewEx.txt', 'r') as file:
    for line in file:
        y_pred_new.append(float(line.strip()))
        
        
        
        
# x = np.linspace(0, 100, 300)       
# # Create a new figure
# plt.figure()

# # Plot the actual values (y) in blue
# plt.plot(x, x_new, label='Actual', color='blue')

# # Plot the predicted values (y_pred) in red
# plt.plot(x, y_pred_new, label='Predicted', color='red')

# # Add labels and a legend
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.legend()

# # Show the plot
# plt.show()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# # Create a new figure
# plt.figure()

# # Plot the actual values (y) in blue
# plt.plot(time, x_old, label='x_old', color='blue')

# # Plot the predicted values (y_pred) in red
# plt.plot(time, y_old, label='y_old', color='red')

# # Add labels and a legend
# plt.xlabel('time')
# plt.ylabel('x and y ')
# plt.legend()

# # Show the plot
# plt.show()




# # Create a new figure
# plt.figure()

# # Plot the actual values (y) in blue
# plt.plot(time, x_new, label='x_new', color='blue')

# # Plot the predicted values (y_pred) in red
# plt.plot(time, y_new, label='y_new', color='red')

# # Add labels and a legend
# plt.xlabel('time')
# plt.ylabel('x and y ')
# plt.legend()

# # Show the plot
# plt.show()





full_x = x_old+ x_new
full_y = y_old + y_new
time2 = np.linspace(0, 100, 600)
# Create a new figure
plt.figure()

# Plot the actual values (y) in blue
plt.plot(time2, full_x, label='x', color='blue')

# Plot the predicted values (y_pred) in red
plt.plot(time2, full_y, label='y', color='red')

# Add labels and a legend
plt.xlabel('time')
plt.ylabel('x and y ')
plt.legend()

# Show the plot
plt.show()

