
# MNIST Dataset Introduction

Most examples are using MNIST dataset of handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255.

## Overview

![MNIST Digits](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)

## Usage

JAX does not include dataset loaders, so the examples use the small helper module [`common/datasets.py`](../common/datasets.py). It handles:

- Dataset downloading (the first time only; files are cached in the `data/` folder).

- Loading the entire dataset into numpy arrays:

```python
from common.datasets import load_mnist

# Load data: uint8 images of shape (n, 28, 28) with values in [0, 255],
# and uint8 labels in [0, 9].
(x_train, y_train), (x_test, y_test) = load_mnist()

# Convert to float32, flatten images to 1-D vector of 784 features (28*28),
# and normalize images value from [0, 255] to [0, 1].
x_train = x_train.reshape([-1, 784]).astype('float32') / 255.
x_test = x_test.reshape([-1, 784]).astype('float32') / 255.
```

- A `batch_iterator` function that shuffles the dataset and iterates over it by batches (in order to only feed a small fraction of the dataset samples to the model at each training step):

```python
from common.datasets import batch_iterator

train_data = batch_iterator(x_train, y_train, batch_size=64)

# Get the next 64 images array and labels.
batch_x, batch_y = next(train_data)
```

Link: http://yann.lecun.com/exdb/mnist/
