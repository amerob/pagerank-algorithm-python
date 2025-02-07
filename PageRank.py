import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm

class PageRank:
    def __init__(self, damping=0.85, max_iter=1000, tol=1e-10):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.ranks_ = None
        self.converged_ = False
        self.n_iter_ = 0
        
    def _create_transition_matrix(self, n, links):
        rows, cols = np.array(links).T if links else (np.array([]), np.array([]))
        data = np.ones_like(rows, dtype=np.float64)
        
        #sparse matrix in CSR format 
        L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        
        col_sums = np.array(L.sum(axis=0)).flatten()
        dangling = col_sums == 0
        col_sums[dangling] = 1  # avoid division by zero
        
        # Normalizion
        return L.multiply(1 / col_sums).tocsc()

    def _power_iteration(self, M, n):
        #power iteration with sparse matrix operations
        ranks = np.full(n, 1/n)
        teleport = (1 - self.damping) / n
        
        for self.n_iter_ in range(1, self.max_iter + 1):
            prev_ranks = ranks.copy()
            ranks = self.damping * M.dot(ranks) + teleport
            delta = spnorm(ranks - prev_ranks, ord=1)
            
            if delta < self.tol:
                self.converged_ = True
                break
                
        return ranks / ranks.sum()  # Ensure proper normalization

    def fit(self, links, n):
        if not 0 < self.damping < 1:
            raise ValueError("Damping factor must be between 0 and 1")
            
        if n <= 0:
            raise ValueError("Number of nodes must be positive")
            
        links = np.asarray(links)
        if links.size > 0 and (links.min() < 0 or links.max() >= n):
            raise ValueError("Invalid node indices in links")

        M = self._create_transition_matrix(n, links)
        self.ranks_ = self._power_iteration(M, n)
        return self

    @property
    def results(self):
        """Return formatted results as percentages"""
        return 100 * (self.ranks_ / self.ranks_.sum())

if __name__ == "__main__":
    links = [
        (0, 1), (0, 2), (1, 0), (2, 0), (2, 1),
        (2, 3), (3, 2), (3, 4), (4, 5), (5, 2)
    ]
    
    pr = PageRank(damping=0.85)
    pr.fit(links, n=6)
    
    print(f"PageRank values: {pr.results.round(2)}%")
    print(f"Converged in {pr.n_iter_} iterations")
    print(f"Convergence status: {pr.converged_}")
