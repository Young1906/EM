from generate_data import generate_data
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from kmean import KMean
from gmm import GMM
from mnist import mnist
parser = ArgumentParser(
        description = "Gaussian Mixture Model")

parser.add_argument(
        "--M",
        type    = int ,
        help    = "Number of underlying Gaussian Distribution")

parser.add_argument(
        "--max_iter",
        type    = int ,
        help    = "Maximum number of iteration")

parser.add_argument(
        "--tol",
        type    = float,
        help    = "Tolerance for termination condition")

parser.add_argument(
        "--init",
        type    = str,
        help    = "Theta initialization methods")

args = parser.parse_args();

if __name__ == "__main__":
    assert args.M > 0, ValueError("Expected M > 0");
    assert args.max_iter > 0, ValueError("Expected max_iter > 0");
    assert args.tol > 0, ValueError("Expected tol > 0");
    assert args.init in {"kmean"}, NotImplementedError(f"{args.init} is not implemented yet");

    gmm = GMM(
            M           = args.M,
            max_iter    = args.max_iter,
            tol         = args.tol,
            init_type   = args.init,
        );

    X, z = generate_data();

    # GMM
    loss = gmm.fit(X);
    label, probs = gmm(X);

    # Kmean
    km = KMean(args.M);
    km.fit(X);

    plt.rcParams["figure.figsize"] = [10, 10]

    fig, axes = plt.subplots(2, 2);

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False);
        ax.spines["right"].set_visible(False);

    axes[0, 0].scatter(X[:,0], X[:, 1], c = km.label, alpha = .5);
    axes[0, 0].set_title("Kmean");

    axes[0, 1].scatter(X[:,0], X[:, 1], c = label, alpha = .5);
    axes[0, 1].set_title("GMM");

    axes[1, 0].scatter(X[:,0], X[:,1], c = z, alpha = .5, )
    axes[1, 0].set_title("Ground Truth");

    axes[1, 1].plot(loss);
    axes[1, 1].set_title("GMM's Loss");

    plt.tight_layout();
    
    fig.savefig("sample.png",)

    

