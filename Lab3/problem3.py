import numpy as np
import scipy as sp
import pylab as pl
from scipy.misc import imread

M = imread('C:\Users\Tony\Downloads\mona_lisa.png', flatten=True)

U, svd, Vt = sp.linalg.svd(M)
approx = sp.linalg.diagsvd(svd, M.shape[0], M.shape[1])
print approx.shape

k_values = [2, 5, 10]

for k in k_values:
    # Question mark...?
    cpy = approx.copy()
    cpy[cpy < svd[int(k)]] = 0
    print svd.shape
    x = np.dot(np.dot(U, cpy), Vt)
    print k
    pl.gray()
    pl.imshow(x)
    #pl.show()
