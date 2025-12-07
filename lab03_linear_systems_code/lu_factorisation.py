import numpy as np

def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L = np.eye(A.shape[0])
    U = np.zeros_like(A)
    
    for j in range(n):
        for i in range(j + 1):
            # Compute factors u_{ij}
            U[i, j] = A[i, j] - np.dot(L[i, :(j + 1)], U[:(j + 1), j])

        for i in range(j + 1, n):
            pass
            # Compute factors l_{ij}
            if U[j, j] == 0:
                raise ZeroDivisionError("One of the diagonal values is zero - cannot do LU factorisation")
            
            L[i, j] = (A[i, j] - np.dot(L[i, :(j + 1)], U[:(j + 1), j])) / U[j, j]
    
    return L, U

# print (lu_factorisation(np.array([[2,1,4], [1,2,2], [2,4,6]])))
# print (lu_factorisation(np.array([[4,2,0], [2,3,1], [0,1,2.5]])))