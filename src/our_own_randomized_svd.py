import numpy as np
from scipy import linalg


def randomized_range_finder(A, size, n_iter=5):
    """
    Compute an orthonormal matrix whose range approximates the range of A.

    https://github.com/scikit-learn/scikit-learn/blob/14031f65d144e3966113d3daec836e443c6d7a5b/sklearn/utils/extmath.py
    power_iteration_normalizer can be safe_sparse_dot (fast but unstable),
    LU (imbetween), or QR (slow but most accurate)
    :param A: 2D array
        The input data matrix
    :param size: integer
        Size of the return array
    :param n_iter: integer
        Number of power iterations used to stabilize the result
    :return: 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.
    """
    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = np.random.normal(size=(A.shape[1], size))

    for i in range(n_iter):
        Q, _ = linalg.lu(A @ Q, permute_l=True)
        Q, _ = linalg.lu(A.T @ Q, permute_l=True)

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(A @ Q, mode="economic")
    return Q


def randomized_svd(M, n_components, n_oversamples=10, n_iter=4):
    """
    Compute a truncated randomized SVD.

    :param M: ndarray or sparse matrix
        Matrix to decompose
    :param n_components: int
        Number of singular values and vectors to extract.
    :param n_oversamples: int
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    :param n_iter: int or 'auto' (default is 'auto')
        Number of power iterations.
    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).
    """
    n_random = n_components + n_oversamples

    Q = randomized_range_finder(M, n_random, n_iter)

    # project M to the (k + p) dimensional space using the basis vectors
    B = Q.T @ M

    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = Q @ Uhat

    return U[:, :n_components], s[:n_components], V[:n_components, :]
