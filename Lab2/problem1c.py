import matplotlib.pyplot as plt
import numpy as np

mean = [0, 0, 0]
cov = [[5, 0, 0],
       [0, 2.48, 0.95],
       [0, 0.95, 20]]

sampleSize = []
covariance = []
for k in range(200):
    numSamples = k * 3000
    sampleSize.append(numSamples)
    samples = np.random.multivariate_normal(mean, cov, numSamples)
    empirical_cov = np.cov(samples.transpose())
    cov_x2x3 = empirical_cov[1][2]
    covariance.append(cov_x2x3)
    
plt.plot(sampleSize, covariance)
#plt.plot((0, .95), (10000, .95), 'k-')
plt.ylabel('Empirical Covariance')
plt.xlabel('Sample size')
plt.show()