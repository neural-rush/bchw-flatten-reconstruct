"""Dataset / tensor preparation. Library calls are allowed here."""
import numpy as np


def load_dataset_samples(B=2, data_dir="./data"):
    """MNIST and CIFAR-10 samples as BCHW float32 arrays.
    Falls back to random tensors of the same shape if torchvision is missing."""
    samples = {}
    try:
        from torchvision import datasets
        mn = datasets.MNIST(data_dir, train=True, download=True)
        x = np.zeros((B, 1, 28, 28), dtype=np.float32)
        for i in range(B):
            img = np.array(mn[i][0], dtype=np.float32) / 255.0      # H x W
            for h in range(28):
                for w in range(28):
                    x[i, 0, h, w] = img[h, w]
        samples["MNIST / sample"] = x

        cf = datasets.CIFAR10(data_dir, train=True, download=True)
        x = np.zeros((B, 3, 32, 32), dtype=np.float32)
        for i in range(B):
            img = np.array(cf[i][0], dtype=np.float32) / 255.0      # H x W x C
            for c in range(3):
                for h in range(32):
                    for w in range(32):
                        x[i, c, h, w] = img[h, w, c]
        samples["CIFAR / sample"] = x
    except Exception as e:
        print(f"[info] torchvision datasets unavailable ({e}); using random stand-ins.")
        rng = np.random.default_rng(0)
        samples["MNIST-like / sample"] = rng.random((B, 1, 28, 28), dtype=np.float32)
        samples["CIFAR-like / sample"] = rng.random((B, 3, 32, 32), dtype=np.float32)
    return samples


def synthetic_fmap(B, C, H, W, seed=0):
    """Synthetic feature map treated as a DNN intermediate activation."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, C, H, W)).astype(np.float32)
