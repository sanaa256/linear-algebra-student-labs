from generate_safe_system import *
from lu_factorisation import *

A_large, b_large, x_large = generate_safe_system(100)

def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U

print(determinant(A_large))