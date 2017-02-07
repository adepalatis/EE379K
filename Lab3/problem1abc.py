import sympy
import numpy as np
from scipy.linalg import orth
from sympy.matrices import Matrix 


v1 = [1,2,3,4]
v2 = [0,1,0,1]
v3 = [1,4,3,6]
v4 = [2,11,6,15]
A = Matrix([v1, v2, v3, v4])
#print np.array(A)

# (1)
# Get the column vectors that span the column space (i.e., span(A))
S = A.columnspace()

# Create a vector in span(A)
# HELP
random_vector = np.random.rand(1, 4)
print random_vector
print np.array(A)
#thing = orth(np.array(A))
#print thing

# Create a vector not in span(A)
# HELP

# (2)
# Calculate dim(S), i.e. the number of vectors in its basis
dim = len(S)


# (3)
# Convert S to a numpy array
S_Array = [[], []] 
for i in range(len(S)):
    for j in range(len(S[0])):
        S_Array[i].append([S[i][j]])
        
#print S_Array
#orthoBasis = orth(A)
