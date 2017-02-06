import numpy as np

# Read the data from CSV, skipping the header row and column
data = np.genfromtxt("C:\Users\Tony\Downloads\DF1", 
                     delimiter=',', 
                     skip_header=1, 
                     usecols=(1,2,3,4))

print np.cov(data,rowvar=False)
print '\n'