import numpy as np
def system_size(A, b):
    """
    Validate the dimensions of a linear system and return its size.

    This function checks whether the given coefficient matrix `A` is square
    and whether its dimensions are compatible with the right-hand side vector
    `b`. If the dimensions are valid, it returns the size of the system.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D array of shape ``(n, n)`` representing the coefficient matrix of
        the linear system.
    b : numpy.ndarray
        A array of shape ``(n, o)`` representing the right-hand side vector.

    Returns
    -------
    int
        The size of the system, i.e., the number of variables `n`.

    Raises
    ------
    ValueError
        If `A` is not square or if the size of `b` does not match the number of
        rows in `A`.
    """

    # Validate that A is a 2D square matrix
    if A.ndim != 2:
        raise ValueError(f"Matrix A must be 2D, but got {A.ndim}D array")

    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A must be square, but got A.shape={A.shape}")

    if b.shape[0] != n:
        raise ValueError(
            f"System shapes are not compatible: A.shape={A.shape}, "
            f"b.shape={b.shape}"
        )

    return n

def forward_substitution(A, b):
    """
    Solve a lower triangular system of linear equations using forward
    substitution.

    This function solves the system of equations:

    .. math::
        A x = b

    where `A` is a **lower triangular matrix** (all elements above the main
    diagonal are zero). The solution vector `x` is computed sequentially by
    solving each equation starting from the first row.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the lower triangular
        coefficient matrix of the system.
    b : numpy.ndarray
        A 1D NumPy array of shape ``(n,)`` or a 2D NumPy array of shape
        ``(n, 1)`` representing the right-hand side vector.

    Returns
    -------
    x : numpy.ndarray
        A NumPy array of shape ``(n,)`` containing the solution vector.
    """
    """
    solves the system of linear equationa Ax = b assuming that A is lower
    triangular. returns the solution x
    """
    # get size of system
    n = system_size(A, b)

    # check is lower triangular
    if not np.allclose(A, np.tril(A)):
        raise ValueError("Matrix A is not lower triangular")

    # create solution variable
    x = np.empty_like(b)

    # perform forwards solve
    for i in range(n):
        partial_sum = 0.0
        for j in range(0, i):
            partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x


def backward_substitution(A, b):
    """
    Solve an upper triangular system of linear equations using backward
    substitution.

    This function solves the system of equations:

    .. math::
        A x = b

    where `A` is an **upper triangular matrix** (all elements below the main
    diagonal are zero). The solution vector `x` is computed starting from the
    last equation and proceeding backward.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the upper triangular
        coefficient matrix of the system.
    b : numpy.ndarray
        A 1D NumPy array of shape ``(n,)`` or a 2D NumPy array of shape
        ``(n, 1)`` representing the right-hand side vector.

    Returns
    -------
    x : numpy.ndarray
        A NumPy array of shape ``(n,)`` containing the solution vector.
    """
    # get size of system
    n = system_size(A, b)

    # check is upper triangular
    assert np.allclose(A, np.triu(A))

    # create solution variable
    x = np.empty_like(b)

    # perform backwards solve
    for i in range(n - 1, -1, -1):  # iterate over rows backwards
        partial_sum = 0.0
        for j in range(i + 1, n):
            partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x