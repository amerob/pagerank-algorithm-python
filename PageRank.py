import numpy as np
import scipy.sparse as sp
import numpy.linalg as la

def create_transition_matrix(n, links):
    data = []
    row_ind = []
    col_ind = []
    for i, j in links:
        data.append(1)
        row_ind.append(i)
        col_ind.append(j)
    L_sparse = sp.coo_matrix((data, (row_ind, col_ind)), shape=(n, n))
    col_sums = np.array(L_sparse.sum(axis=0)).flatten()
    L_sparse = L_sparse.multiply(1.0 / col_sums)
    return L_sparse

def power_iteration(L_sparse, d, max_iter=100, tol=1e-6):
    n = L_sparse.shape[0]
    r = np.ones(n) / n
    J = sp.coo_matrix(np.ones((n, n)) / n)
    M = d * L_sparse + (1 - d) * J
    for _ in range(max_iter):
        last_r = r
        r = M @ r
        if la.norm(r - last_r) < tol:
            break
    return r

def pageRank(links, n, d=0.85, max_iter=100, tol=1e-6):
    L_sparse = create_transition_matrix(n, links)
    r = power_iteration(L_sparse, d, max_iter, tol)
    return 100 * np.real(r / np.sum(r))

links = [
    (0, 1), (0, 2), (1, 0), (2, 0), (2, 1),
    (2, 3), (3, 2), (3, 4), (4, 5), (5, 2)
]

n = 6
d = 0.85
pageRank_values = pageRank(links, n, d)
print(f"PageRank values: {pageRank_values}")
