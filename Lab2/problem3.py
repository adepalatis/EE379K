import numpy as np
import math

# Declare constants
beta_naught = -3
beta = 0
n = 150
numSamples = 2000

distances = []
for x in range(numSamples):
    x = np.random.normal(0, 1, n)
    e = np.random.normal(0, 1, n)
    
    # Generate y_i = beta_naught + e_i
    y = beta_naught + x * beta + e
    
    # Calculate beta_hat for the dataset
    beta_hat = np.dot(x, y) / np.dot(x, x)
    #beta_hat = (x.transpose() * y) * (x.transpose() * x)
    sd = beta_hat - beta
    distances.append(sd)
    
sum = 0
for num in distances:
    sum += math.pow(num, 2)
std_dev = math.sqrt(sum / numSamples)
print std_dev
