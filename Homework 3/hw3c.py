import numpy as np


def is_symmetric(A):
    return np.allclose(A, A.T)


def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def cholesky_solve(A, b):
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x, "Cholesky Method"


def doolittle_solve(A, b):
    n = len(A)
    L = np.eye(n)
    U = np.zeros_like(A)

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x, "Doolittle Method"


def solve_system(A, b):
    if is_symmetric(A) and is_positive_definite(A):
        return cholesky_solve(A, b)
    else:
        return doolittle_solve(A, b)


if __name__ == "__main__":
    A1 = np.array([[6, 3, 4], [3, 6, 5], [4, 5, 10]], dtype=float)
    b1 = np.array([1, 2, 3], dtype=float)

    A2 = np.array([[4, 2, 0], [2, 4, 2], [0, 2, 3]], dtype=float)
    b2 = np.array([4, 6, 7], dtype=float)

    x1, method1 = solve_system(A1, b1)
    x2, method2 = solve_system(A2, b2)

    print(f"Solution for A1 using {method1}: {x1}")
    print(f"Solution for A2 using {method2}: {x2}")
