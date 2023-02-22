import gzip, os
import numpy as np
# from matplotlib import pyplot as plt

def mnist(
        subset="train",
    )->np.ndarray:
    """
    Load MNIST from local folder & emulate missing value condition
    If local folder not exists, run

    ```bash
    make download_mnist
    ```
    """

    assert subset in ("train", "test"), \
            ValueError("Expected subset in (train, test)");

    if subset == "train":
        label_fn = "train-label.gz"
        image_fn = "train-image.gz"

    else:
        label_fn = "test-label.gz"
        image_fn = "test-image.gz"

    # Load image
    with gzip.open(f"datasets/{image_fn}") as f:
        img = np.frombuffer(f.read(), "B", offset=16);
        img = img.reshape(-1, 784).astype("float32");
    
    # Load label
    with gzip.open(f"datasets/{label_fn}") as f:
        lab = np.frombuffer(f.read(), "B", offset=8);

    return img, lab

