import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from pandas.tools.plotting import scatter_matrix

# Read data from CSV
data = df.from_csv(path="C:\Users\Tony\Downloads\DF2")

# Get the mean and covariance of the data
mean = data.mean()
cov = data.cov()

# Find the inverse of the covariance matrix
cov_inv = pd.DataFrame(np.linalg.pinv(cov.values), cov.columns, cov.index)

# Multiply the identity matrix by the result
idMatrix = df(np.identity(2))
transFactor = np.dot(idMatrix, cov_inv)

# Plot the untransformed data
data.columns = ['a', 'b']
plt.scatter(data.a, data.b)
plt.show()

# Transform the data and plot again
data = np.dot(data, transFactor)
plt.scatter(data[:,0], data[:,1])
plt.show()
