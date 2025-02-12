import math


def is_symmetric(A):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    return True


def is_positive_definite(A):
    n = len(A)
    for i in range(1, n + 1):
        sub_matrix = [row[:i] for row in A[:i]]
        if determinant(sub_matrix) <= 0:
            return False
    return True


def determinant(matrix):
    """Computes the determinant of a square matrix using recursion."""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i in range(n):
        minor = [row[:i] + row[i + 1:] for row in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * determinant(minor)
    return det


def cholesky_solve(A, b):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]

    y = forward_substitution(L, b)
    x = backward_substitution([[L[j][i] for j in range(n)] for i in range(n)], y)
    return x, "Cholesky Method"


def doolittle_solve(A, b):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x, "Doolittle Method"


def forward_substitution(L, b):
    n = len(L)
    y = [0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def solve_system(A, b):
    if is_symmetric(A) and is_positive_definite(A):
        return cholesky_solve(A, b)
    else:
        return doolittle_solve(A, b)


if __name__ == "__main__":
    A1 = [[6, 3, 4], [3, 6, 5], [4, 5, 10]]
    b1 = [1, 2, 3]

    A2 = [[4, 2, 0], [2, 4, 2], [0, 2, 3]]
    b2 = [4, 6, 7]

    x1, method1 = solve_system(A1, b1)
    x2, method2 = solve_system(A2, b2)

    print(f"Solution for A1 using {method1}: {x1}")
    print(f"Solution for A2 using {method2}: {x2}")
