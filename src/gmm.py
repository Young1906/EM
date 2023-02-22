"""
"""
import numpy as np
from kmean import KMean

def abs(x):
    return x if x>=0 else -x;

class GMM:
    def __init__(self,
            M           : int               ,   # number of underlying Gaussian distribution
            max_iter    : int               ,   # number of maximum iteration
            tol         : float = 1e-3      ,   # stopping condition
            init_type   : str   = "kmean"   ,   # theta initialization
            ):

        self.M          = M;
        self.max_iter   = max_iter;
        self.tol        = tol;
        self.init_type  = init_type;

        # \theta = (\pi, \mu, \Sigma)
        self.Pi         = None;
        self.Mu         = None;
        self.Sigma      = None;

        # Check if parameter if initialized at t=0;
        self.inited     = False;
        self.__fitted     = False;

    def __init(self, X):
        if self.init_type == "kmean":
            self.__init_kmean(X);
        else:
            raise NotImplementedError(f"Method {self.init_type} is not implemented");


    def __init_kmean(self, X):
        """
        Using KNN to initiate Pi, Mu, Sigma
        """
        if self.inited:
            return;
        
        assert X.ndim == 2, ValueError("Expected a 2-D numpy array");

        # X's dim
        _, P = X.shape

        km = KMean(self.M);
        km.fit(X);

        # getting centroid & scale
        self.Mu = km.centroids;

        # Initializing Sigma (full covariance matrix)
        self.Sigma = np.zeros((self.M, P, P));

        for i in range(self.M):
            idx_i, = np.where(km.label == i);
            X_i = X[idx_i, :];

            # Covariance of cluster ith
            cov_i = np.cov(X_i, rowvar = False, bias = False);
            self.Sigma[i, :, :] = cov_i;

        # Pi
        self.Pi = np.zeros(self.M);

        for i in range(self.M):
            self.Pi[i] = (km.label == i).mean();


        # Initial param is inited
        self.inited     = True;


    def fit(self, X):
        """
        """

        # Initializing Pi, Mu, Sigma
        self.__init(X);

        # Terminating condition
        prev_loss = None;
        loss = [];

        for i in range(self.max_iter):
            # Update theta + return membership at time
            loss_i = self.__step(X);
            loss.append(loss_i);

            # Terminating condition
            if prev_loss is None:
                prev_loss = loss_i;
                continue;

            # Early terimnation
            if abs(loss_i - prev_loss) < self.tol:
                break;

            prev_loss = loss_i;

        # If GMM have been fitted;
        self.__fitted = True;

        return np.array(loss);


    def __call__(self, X):
        """
        Inference step
        """

        assert self.__fitted, Exception("Model has not been fitted yet bro!!!");

        if X.ndim == 1:
            X = X.expand_dims(0);

        N, P = X.shape;

        label_ = np.zeros(N);
        probs = np.zeros((N, self.M));

        for i in range(N):
            x_i = X[i,:];
            for k in range(self.M):
                probs[i, k] = np.exp(self.log_conditional_pdf(
                        x_i,
                        k,
                        self.Mu,
                        self.Sigma))

            label_[i] = np.argmax(probs[i,:]);

        return label_, probs




    def __step(self, X):
        """
        """

        N, P = X.shape;

        # E step : compute the membership probability matrix
        A = np.zeros((N, self.M));

        for i in range(N):
            x_i = X[i,:];
            for k in range(self.M):
                A[i, k] = self.Pi[k] * \
                        np.exp(self.log_conditional_pdf(x_i, 
                                k,
                                self.Mu,
                                self.Sigma))

        A = A / A.sum(-1)[:, np.newaxis];

        # M step: update Pi, Mu, Sigma
        Pi = np.zeros(self.M);
        Mu = np.zeros((self.M, P));
        Sigma = np.zeros((self.M, P, P));

        for k in range(self.M):
            # \pi_k  = \sum_{i=1}^N {A_ik} / N
            Pi[k] = A[:, k].sum() / N;

            # \mu_k = \sum_{i=1}^N{A_ik * x_i} / \sum_{i=1}^N{A_ik}
            Mu[k] = (X * A[:, k][:, np.newaxis]).sum(0) / A[:, k].sum()
            # import pdb; pdb.set_trace();

            # Sigma
            nominator = np.zeros((P, P));
            for i in range(N):
                x_i = X[i, :]
                nominator += A[i, k] * np.outer(x_i - Mu[k], x_i - Mu[k]);

            Sigma[k, :, :] = nominator / A[:, k].sum()

        # Loss at time t
        loss = self.Q(X, A);

        # Update theta
        # self.Pi = Pi;

        # Let pi = softmax(pi)
        self.Pi = np.exp(Pi) / np.exp(Pi).sum(-1)
        self.Mu = Mu;
        self.Sigma = Sigma;

        return loss;


    def Q(self, X, A):
        """
        Surrogate cost function E_z|x,theta^t[logP(X,Z|theta^t]
        """
        N, _ = X.shape;

        out = 0;
        for i in range(N):
            x_i = X[i,:]
            for k in range(self.M):
                out += A[i, k] * (np.log(self.Pi[k]) + \
                        self.log_conditional_pdf(
                            x_i,
                            k,
                            self.Mu,
                            self.Sigma,
                        ))
        return out;
        

    @staticmethod
    def log_conditional_pdf(
            x           : np.ndarray    , 
            k           : int           ,
            Mu_t        : np.ndarray    , 
            Sigma_t     : np.ndarray    , 
            )->float:   
        """
        Return multivariate normal p.d.f. value for vector x
        """

        assert Mu_t.ndim == 2, ValueError("Wrong location's shape");
        assert Sigma_t.ndim == 3, ValueError("Wrong scale's shape");

        mu_tk = Mu_t[k, :];             # The kth component of mu at time t
        sigma_tk = Sigma_t[k, :, :]     # the kth covariance at time t

        # shape of x should be a 1xP vector
        x = np.squeeze(x);
        assert x.ndim == 1, ValueError("Expected a vector!!!");
        P, = x.shape;

        log_prob = -0.5 * P * np.log(2 * np.pi) \
               -0.5 * np.log(np.linalg.det(sigma_tk))\
               -.5 * (x - mu_tk) @ np.linalg.inv(sigma_tk) @ (x - mu_tk).T;

        return log_prob / P




if __name__ == "__main__":
    x = np.random.uniform(-1,1,2);

    Mu = np.zeros((2,2));
    Sigma = np.stack([np.diag(np.ones(2)) for _ in range(2)])

    print(Mu.shape, Sigma.shape)
    print(GMM.conditional_pdf(x, 0, Mu, Sigma))

    var = multivariate_normal(Mu[0,:], Sigma[0,:,:])
    print(var.pdf(x))
