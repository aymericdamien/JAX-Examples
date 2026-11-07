""" Nearest Neighbor Example.

A nearest neighbor learning algorithm example using JAX library: to classify
a test image, find the closest training image (using the L1 distance) and
use its label.

This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_mnist

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = load_mnist()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# %%
# In this example, we limit mnist data.
Xtr, Ytr = x_train[:5000], y_train[:5000]  # 5000 for training (nn candidates)
Xte, Yte = x_test[:200], y_test[:200]  # 200 for testing

# Nearest Neighbor calculation using L1 Distance.
# 'xtr' holds all the training images, 'xte' is a single test image.
@jax.jit
def nearest_neighbor(xtr, xte):
    # Calculate L1 Distance between the test image and every training image.
    distance = jnp.sum(jnp.abs(xtr - xte), axis=1)
    # Prediction: Get min distance index (Nearest neighbor).
    return jnp.argmin(distance)

accuracy = 0.

# %%
# Loop over test data.
for i in range(len(Xte)):
    # Get nearest neighbor.
    nn_index = nearest_neighbor(Xtr, Xte[i])
    # Get nearest neighbor class label and compare it to its true label.
    print("Test", i, "Prediction:", Ytr[nn_index], "True Class:", Yte[i])
    # Calculate accuracy.
    if Ytr[nn_index] == Yte[i]:
        accuracy += 1. / len(Xte)
print("Done!")
print("Accuracy:", accuracy)

# %%
# Process all test images at once with jax.vmap.
# 'nearest_neighbor' is written for a single test image. jax.vmap turns it
# into a function processing a batch of test images: in_axes=(None, 0) means
# 'xtr' is shared, and 'xte' is mapped over its first dimension.
batch_nearest_neighbor = jax.vmap(nearest_neighbor, in_axes=(None, 0))
nn_indices = batch_nearest_neighbor(Xtr, Xte)
print("Accuracy (vmap):", np.mean(Ytr[nn_indices] == Yte))
