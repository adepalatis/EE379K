import numpy as np
import matplotlib.pyplot as plt

# n = 5
sampleMeans = []
for x in range(1000):
    samples = np.random.binomial(1, 0.5, 5)
    sampleMeans.append(float(sum(samples)) / 5)
    
plt.hist(sampleMeans, bins=30)
plt.show()

# n = 300
sampleMeans = []
for x in range(1000):
    samples = np.random.binomial(1, 0.5, 300)
    sampleMeans.append(float(sum(samples)) / 300)

plt.hist(sampleMeans, bins=30)
plt.show()