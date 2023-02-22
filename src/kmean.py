import numpy as np
import matplotlib.pyplot as plt

class KMean:
    def __init__(self,
            K           : int   = 2,
            max_iter    : int   = 100,
            tol         : float = 1e-4):
        self.K = K;

        # store location of the centroid
        self.centroids = None;

        # store label of the obs
        self.label = None;
        self.is_inited = False;

        # Other params
        self.max_iter = max_iter;
        self.tol = tol;

    def __init(self, X):
        if self.is_inited:
            return

        N, P = X.shape;
        
        self.label = np.random.randint(0, self.K, N);
        self.centroids = np.zeros((self.K, P));
        self.update_centroid(X);

        self.is_inited = True

    def update_centroid(self, X):
        """
        Update centroid
        """
        for i in range(self.K):
            idx, = np.where(self.label == i);
            _X = X[idx, :]
            self.centroids[i,:] = np.mean(_X);



    def fit(self, X):
        self.__init(X);
        N, P = X.shape;

        prev_loss = self.loss(X);

        for i in range(self.max_iter):
            self.take_one_step(X);
            loss = self.loss(X);

            if prev_loss - loss < self.tol:
                break;

            prev_loss = loss;




    def closest_centroid(self, x):
        """
        Return the closest centroid to x
        """
        dist = self.centroids - x[np.newaxis,:]
        dist = (dist * dist).sum(-1)
        return np.argmin(dist);

    def take_one_step(self, X):
        N, _ = X.shape;

        for i in range(N):
            closest_centroid = self.closest_centroid(X[i,:]);
            self.label[i] = closest_centroid;

        self.update_centroid(X);

    def loss(self, X):
        loss = 0;

        for i in range(self.K):
            # Calculate distance from centroid i to all of its member
            idx, = np.where(self.label == i);
            X_ = X[idx, :];
            centroid = self.centroids[i, :];

            dist = X_ - centroid[np.newaxis, :];
            dist = (dist * dist).sum(-1);
            loss += dist.sum();
        return loss;



if __name__ == "__main__":
    X0 = np.random.multivariate_normal(
            np.array([0, 0]),
            np.array([[1, 0], [0, 1]]),
            100);
    X1 = np.random.multivariate_normal(
            np.array([0, 3]),
            np.array([[1, 0], [0, 1]]),
            100);
    
    X2 = np.random.multivariate_normal(
            np.array([3, 3]),
            np.array([[1, 0], [0, 1]]),
            100);

    ground_truth = np.array([*[0]*100, *[1]*100, *[2]*100  ])
    X = np.concatenate([X0,X1,X2]);
    

    km = KMean(3);
    km.fit(X)

    fig, axes = plt.subplots(1,2)

    axes[0].scatter(X[:, 0], X[:, 1], c=ground_truth);
    axes[0].set_title("Ground Truth");

    axes[1].scatter(X[:, 0], X[:, 1], c=km.label);
    axes[1].set_title("KMean");

    plt.show();




