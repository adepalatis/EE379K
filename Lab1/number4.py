import numpy as np
import math
import matplotlib.pyplot as plt

mean = [-5, 5]
cov = [[20, 0.8], [0.8, 30]]

samples = np.random.multivariate_normal(mean, cov, 10000)

sums = np.sum(samples, axis=0)
sampleMean = sums / len(samples)
#print np.mean(samples, axis=0)
   
varX = 0
varY = 0
sampleCov = 0
for x in range(len(samples)):
    sampleCov += (samples[x][0] - sampleMean[0]) * (samples[x][1] - sampleMean[1])     #calculate cov(x,y)
    varX += math.pow((samples[x][0] - sampleMean[0]), 2)     # calculate var(x)
    varY += math.pow((samples[x][1] - sampleMean[1]), 2)     # calculate var(y)
sampleCov /= len(samples)
varX /= len(samples)
varY /= len(samples)

covMatrix = np.array([[varX, sampleCov], [sampleCov, varY]])
#print np.cov(samples.T)

# Print results
print "Sample Mean Matrix: " #+ str(sampleMean)
print sampleMean
print "Sample Covariance Matrix: " #+ str(covMatrix)
print covMatrix
