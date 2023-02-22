"""
    Gaussian Mixture Model implementation
    ---
    Author
    - Tu T. Do
"""

import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    """
    """
    def __init__(self,
            M           : int,              # Number of
            P           : int,              # Input X's dimension
            tol         : float = 1e-3,
            max_iter    : int   = 100,
        ):             
        """
        Args:
        - M: number of underlying Gaussian distributions
        - tol
        - max_iter
        """
        self.M, self.tol, self.P, self.max_iter = M, tol, P, max_iter

        # Mixture weight, initialized uniformly
        self.Pi = np.array([1./M] * M)

        # Location
        self.Mu = np.random.normal(0, 1, (M, P))

        # Initializing scale
        V = [np.random.uniform(0, 1, (P, P)) for _ in range(M)]
        Sigma = [ v @ v.T for v in V ]
        self.Sigma = np.stack(Sigma);

        # Clean up
        del Sigma, V

    def f(self, x, k):
        mu_k = self.Mu[k,:]
        Sigma_k = self.Sigma[k, :, :]

        var = multivariate_normal(mu_k, Sigma_k);

        return var.pdf(x);


    def I(p : bool) ->int:
        return 1 if p else 0;

    def loglikelihood(self, X):
        N, _ = X.shape;

        l = 0;
        for i in range(N):
            for k in range(self.M):
                l += np.log(self.Pi[k]) + np.log(self.f(X[i,:], k))
        return l;


    def step(self, X):
        assert X.ndim == 2;

        # E-step
        # Calculate membership weight
        N, _ = X.shape

        A = np.zeros((N, self.M))

        for i in range(N):
            for k in range(self.M):
                A[i,k] = self.Pi[k] * self.f(X[i,:], k)

        A = A / np.sum(A, -1)[:, np.newaxis]

        # Mstep : 
        # Update pi
        for k in range(self.M):
            self.Pi[k] = A[:, k].sum()/N

        # Update \mu
        for k in range(self.M):
            self.Mu[k, :] = (A[:, k][:, np.newaxis] * X).sum() / A[:, k].sum()

        # Update \Sigma
        for k in range(self.M):
            nominator = 0;
            for i in range(N):
                nominator += A[i,k] * np.outer(
                        X[i,:] - self.Mu[k,:],
                        X[i,:] - self.Mu[k,:]
                        )
            
            self.Sigma[k,:, :] = nominator / A[:, k].sum()
            # import pdb; pdb.set_trace()

    def fit(self, X):
        loss = [];
        for i in range(self.max_iter):
            print(i)
            self.step(X)
            l_i = self.loglikelihood(X);
            print(i, l_i)
            # if l_i - loss[-1] < self.tol: break
            # loss.append(l_i)


