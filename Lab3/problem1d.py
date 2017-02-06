import numpy as np
from scipy import optimize
from sympy.matrices import Matrix

def min_func(x, z, guess):
    #print np.linalg.norm(x - z)
    return np.linalg.norm(x - z)

def main():
    z_star = [1,0,0,0]
    v1 = [1,2,3,4]
    v2 = [0,1,0,1]
    v3 = [1,4,3,6]
    v4 = [2,11,6,15]
    A = Matrix([v1, v2, v3, v4])
    S = A.columnspace()
    print np.random.rand(4, 1)
    
    res = optimize.fmin(min_func, np.random.rand(4, 1), args=(np.random.random((1, 4))[0], z_star))
    print res

    
if __name__ == "__main__":
    main()