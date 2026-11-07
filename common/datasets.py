""" Datasets used by the JAX examples.

JAX does not ship dataset loaders, so this small module provides them:
every `load_*` function downloads a dataset (once), caches it under 'data/'
at the repository root, and returns it as plain NumPy arrays: the training
and test sets, each as a pair (inputs, labels).

    from common.datasets import load_mnist
    (x_train, y_train), (x_test, y_test) = load_mnist()

It also provides `batch_iterator`, a minimal "shuffle and batch" data
loader.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import gzip
import os
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np

# Datasets are cached under 'data/' at the repository root (override it
# with the JAX_EXAMPLES_DATA environment variable).
DATA_DIR = os.environ.get(
    "JAX_EXAMPLES_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def download(urls, filename):
    """Download 'filename' into DATA_DIR (from the first working mirror)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    for url in urls:
        try:
            print("Downloading %s ..." % url)
            # Write to a '.part' file first, so that an interrupted download
            # never leaves a truncated file behind.
            with urllib.request.urlopen(url, timeout=60) as response, \
                    open(path + ".part", "wb") as f:
                shutil.copyfileobj(response, f)
            os.replace(path + ".part", path)
            return path
        except Exception as e:
            print("  failed (%s), trying next mirror." % e)
    raise RuntimeError("Could not download %s." % filename)


def save_cache(path, arrays):
    """Save NumPy arrays to a '.npz' cache file, through a '.part' file."""
    with open(path + ".part", "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(path + ".part", path)


# ---------------------------------------------------------------------------
# MNIST: 70,000 28x28 grayscale images of handwritten digits.
# ---------------------------------------------------------------------------

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]


def _read_idx(path):
    # MNIST files are gzipped 'IDX' files: a small header, then raw uint8.
    with gzip.open(path, "rb") as f:
        data = f.read()
    # Header: 2 zero bytes, the data type (0x08 = uint8), the number of
    # dimensions, then the size of each dimension (big-endian uint32).
    if data[:3] != b"\x00\x00\x08":
        raise ValueError("%s is not a valid MNIST file, delete it and retry." % path)
    ndim = data[3]
    shape = np.frombuffer(data, ">u4", count=ndim, offset=4)
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist():
    """Load the MNIST dataset.

    Returns:
        (x_train, y_train), (x_test, y_test): uint8 images of shape
        (n, 28, 28) with values in [0, 255], and uint8 labels in [0, 9].
        There are 60,000 training and 10,000 test samples.
    """
    cache = os.path.join(DATA_DIR, "mnist.npz")
    if not os.path.exists(cache):
        files = {"x_train": "train-images-idx3-ubyte.gz",
                 "y_train": "train-labels-idx1-ubyte.gz",
                 "x_test": "t10k-images-idx3-ubyte.gz",
                 "y_test": "t10k-labels-idx1-ubyte.gz"}
        arrays = {}
        for name, filename in files.items():
            path = download([m + filename for m in MNIST_MIRRORS], filename)
            arrays[name] = _read_idx(path)
            os.remove(path)
        save_cache(cache, arrays)
    with np.load(cache) as data:
        return ((data["x_train"], data["y_train"]),
                (data["x_test"], data["y_test"]))


# ---------------------------------------------------------------------------
# CIFAR-10: 60,000 32x32 color images in 10 classes.
# ---------------------------------------------------------------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def load_cifar10():
    """Load the CIFAR-10 dataset.

    Returns:
        (x_train, y_train), (x_test, y_test): uint8 images of shape
        (n, 32, 32, 3) with values in [0, 255], and uint8 labels in [0, 9]
        of shape (n, 1). There are 50,000 training and 10,000 test samples.
    """
    cache = os.path.join(DATA_DIR, "cifar10.npz")
    if not os.path.exists(cache):
        path = download([CIFAR10_URL], "cifar-10-binary.tar.gz")
        splits = {"train": ["data_batch_%d.bin" % i for i in range(1, 6)],
                  "test": ["test_batch.bin"]}
        arrays = {}
        with tarfile.open(path) as tar:
            for split, names in splits.items():
                raw = b"".join(
                    tar.extractfile("cifar-10-batches-bin/" + name).read()
                    for name in names)
                # Each record: 1 label byte, then a 3x32x32 image (CHW).
                records = np.frombuffer(raw, np.uint8).reshape(-1, 3073)
                arrays["y_" + split] = records[:, 0]
                arrays["x_" + split] = (records[:, 1:].reshape(-1, 3, 32, 32)
                                        .transpose(0, 2, 3, 1))  # To HWC.
        save_cache(cache, arrays)
        os.remove(path)
    with np.load(cache) as data:
        # Labels have shape (n, 1): one row per image.
        return ((data["x_train"], data["y_train"].reshape(-1, 1)),
                (data["x_test"], data["y_test"].reshape(-1, 1)))


# ---------------------------------------------------------------------------
# Boston Housing: 506 houses, 13 features each, and their median value.
# ---------------------------------------------------------------------------

BOSTON_URL = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
              "boston_housing.npz")


def load_boston_housing(test_split=0.2, seed=113):
    """Load the Boston Housing dataset, with a fixed train/test split.

    Returns:
        (x_train, y_train), (x_test, y_test): float64 features of shape
        (n, 13), and house values in thousands of dollars (float64).
        The split is the same at every run (404 train, 102 test samples).
    """
    path = download([BOSTON_URL], "boston_housing.npz")
    with np.load(path, allow_pickle=False) as data:
        x, y = data["x"], data["y"]
    indices = np.arange(len(x))
    np.random.RandomState(seed).shuffle(indices)
    x, y = x[indices], y[indices]
    n_train = int(len(x) * (1 - test_split))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


# ---------------------------------------------------------------------------
# text8: the first 100M characters of an English Wikipedia dump.
# ---------------------------------------------------------------------------

TEXT8_MIRRORS = [
    "https://mattmahoney.net/dc/text8.zip",
    "https://data.deepai.org/text8.zip",
]


def load_text8():
    """Load the text8 Wikipedia corpus, as a list of ~17M lowercase words."""
    path = download(TEXT8_MIRRORS, "text8.zip")
    with zipfile.ZipFile(path) as f:
        text = f.read(f.namelist()[0]).decode("ascii")
    return text.split()


# ---------------------------------------------------------------------------
# A minimal data loader.
# ---------------------------------------------------------------------------

def batch_iterator(x, y, batch_size, seed=0):
    """Shuffle and batch (x, y) forever, epoch after epoch.

    Every epoch, the samples are shuffled and split into batches. The last
    incomplete batch of each epoch is dropped: all batches having the same
    shape avoids re-compiling the jitted training step.
    """
    rng = np.random.default_rng(seed)
    num_batches = len(x) // batch_size
    if num_batches == 0:
        raise ValueError("batch_size (%d) is larger than the dataset (%d samples)."
                         % (batch_size, len(x)))
    while True:
        indices = rng.permutation(len(x))
        for i in range(num_batches):
            batch = indices[i * batch_size:(i + 1) * batch_size]
            yield x[batch], y[batch]
