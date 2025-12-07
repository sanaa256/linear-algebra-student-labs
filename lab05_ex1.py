import numpy as np

# Gram-Schmidt function from lecture notes
def gram_schmidt_qr(A):
    """
    Compute the QR factorisation of a square matrix using the classical
    Gram-Schmidt process.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix.

    Returns
    -------
    Q : numpy.ndarray
        Orthonormal matrix of shape ``(n, n)`` where the columns form an
        orthonormal basis for the column space of A.
    R : numpy.ndarray
        Upper triangular matrix of shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        # Start with the j-th column of A
        u = A[:, j].copy()

        # Orthogonalize against previous q vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # projection coefficient
            u -= R[i, j] * Q[:, i]  # subtract the projection

        # Normalize u to get q_j
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R


def compute_A(epsilon):
    A = np.array([[1, 1 + epsilon], [1 + epsilon, 1]])
    return A
    
def error_1(A, Q, R):
    QR = Q @ R
    error = np.linalg.norm(np.subtract(A, QR))
    return error

def error_2(Q, n):
    I = np.eye(n)
    QT = Q.transpose()
    QTQ = QT @ Q
    error = np.linalg.norm(np.subtract(QTQ, I))
    return error

def error_3(R):
    error = np.linalg.norm(np.subtract(R, np.triu(R)))

columns = []
epsilon = 0
epsilon_vals = [10 ** i for i in range(-6, -17, -1)]

heading_format = "{:<8}| {:<23}| {:<23}| {:<23}"
row_format = "{:<8}| {:<23}| {:<23}| {:<23}"
print(heading_format.format("ε", "error_1", "error_2", "error_3"))


for val in epsilon_vals:
    A = compute_A(val)
    Q, R = gram_schmidt_qr(A)
    print(row_format.format(val, error_1(A, Q, R), error_2(Q, 2), str(error_3(R))))

