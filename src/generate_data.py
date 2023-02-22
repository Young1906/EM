import numpy as np


def generate_data():
    """
    """
    x = np.random.normal(0, 1, 100);
    y0 = 2 * x + 1 + np.random.normal(0, .5, 100);
    y1 = -.5 * x + 2 + np.random.normal(0, .5, 100);

    x0 = np.stack([x, y0]);
    x1 = np.stack([x, y1]);

    lab = np.array([*[0] * 100, *[1]*50])



    return np.concatenate([x0.T, x1.T[:50,:]], axis=0), lab


