import sympy
import numpy as np
from scipy.linalg import orth
from sympy.matrices import Matrix 

# Projects vector 'b' onto vector 'a'
def proj(b, a):
    return (np.dot(a, b) / np.linalg.norm(a) ** 2) * a

v1 = [1,2,3,4]
v2 = [0,1,0,1]
v3 = [1,4,3,6]
v4 = [2,11,6,15]
A = Matrix([v1, v2, v3, v4])
S = np.array(A.rref()[0]).astype(np.float)     # S is a basis for {v1, v2, v3, v4}

# (1)
# Generate a random 1x4 vector
random_vector = np.random.rand(1, 4)[0]

# Generate an orthoganal basis for S, then project 'random_vector' onto each of them
orth_comp = S[0] - proj(S[0], S[1])    # the component of S[0] orthogonal to S[1]
vec_in_space = proj(random_vector, orth_comp) + proj(random_vector, S[1])
#test = Matrix([Matrix(S), vec_in_space])
#print np.array(test)
#print np.array(test.rref()[0])
print 'New vector inside S: ' + str(vec_in_space)

# Create a vector not in span(A)
vec_not_in_space = np.random.rand(1, 4)[0]
#test = Matrix([Matrix(S), vec_not_in_space])
#print np.array(test)
#print np.array(test.rref()[0])
print 'New vector not in S: ' + str(vec_not_in_space)

# (2)
# Calculate dim(S), i.e. the number of vectors in its basis
dim = len(A.columnspace())
print 'Dim(S): ' + str(dim)

# (3)
asArray = np.array(A).astype(np.float)
thing = orth(asArray)

vec1 = []
vec2 = []
for i in range(len(thing)):
    vec1.append(thing[i][0])
    vec2.append(thing[i][1])
    
vec1 = np.array(vec1)
vec1 = np.array(vec1).astype(np.float)
ortho_basis = np.array((vec1, vec2))
print 'Orthonormal basis for S: ' + str(ortho_basis)

# (4)
z_star = [1,0,0,0]

random_vector = np.random.rand(1, 4)[0]
v0 = vec_in_space
v1 = proj(random_vector, orth_comp) + proj(random_vector, S[1])

norm_v0 = np.linalg.norm(v0) 
norm_v1 = np.linalg.norm(v1) 

c0 = (1 / (norm_v0**2)) * (np.dot(v0.transpose(), z_star))
c1 = (1 / (norm_v0**2)) * (np.dot(v1.transpose(), z_star))
print c0
print c1