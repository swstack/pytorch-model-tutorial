import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate random input data (x)
x = np.round(np.random.uniform(low=0, high=20, size=100), decimals=2)

# Generate random noise for the output data
noise = np.random.normal(loc=0, scale=3, size=100)

# Generate output data (y) based on a linear relationship with noise
# Let's assume the true relationship is y = 3x + 3
y = np.round(3 * x + 3 + noise, decimals=2)

# Perform linear regression to find the slope and intercept of the line
slope, intercept = np.polyfit(x, y, 1)

# Generate points for the linear regression line
x_line = np.linspace(0, 20, 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='red', label='Linear Regression Line')

plt.scatter(x, y, label='Data points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Dataset')
plt.legend()
plt.grid(True)
plt.savefig("data.png", transparent=None, dpi='figure', format=None,
            metadata=None, bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto', backend=None)

data = np.column_stack((x, y))
np.savetxt("data.csv", data, fmt="%.2f", delimiter=',')
