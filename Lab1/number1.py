from numpy import random
import numpy as np
import matplotlib.pyplot as plt

first, second, sum = ([] for i in range(3))
first = random.normal(-10, 5, 1000)
second = random.normal(10, 5, 1000)

sum = first + second

# Get mean and variance of sum
mean = np.mean(sum)
var = np.var(sum)
print mean
print var

# Plot the histogram of the data
plt.hist(sum, bins=30)
plt.show()


