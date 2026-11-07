""" K-Means Example.

Implement K-Means algorithm with JAX, and apply it to classify handwritten
digit images. This example is using the MNIST database of handwritten digits
as training samples (http://yann.lecun.com/exdb/mnist/).

K-Means groups the images into k clusters: each image is assigned to its
closest centroid, then each centroid moves to the mean of its images, and
these two steps are repeated. Each centroid then gets the most frequent
label of its images, to classify new images.

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

# %%
# Parameters.
num_steps = 50  # Total steps to train.
k = 25  # The number of clusters.
num_classes = 10  # The 10 digits.
num_features = 784  # Each image is 28x28 pixels.

# %%
# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = load_mnist()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.
full_data_x = x_train

# %%
# Initialize the centroids with k images picked at random in the training set.
# JAX random functions take a random 'key' as input.
key = jax.random.key(0)
centroids = jax.random.choice(key, full_data_x, shape=[k], replace=False)

# %%
# Cosine distance between every image and every centroid.
def cosine_distance(x, centroids):
    # Normalize every vector to a length of 1.
    x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    centroids = centroids / jnp.linalg.norm(centroids, axis=1, keepdims=True)
    # Distance = 1 - cosine similarity. Output shape: [num_images, k].
    return 1. - jnp.matmul(x, centroids.T)

# Assign each image to its closest centroid.
def assign_clusters(centroids, x):
    distances = cosine_distance(x, centroids)
    cluster_idx = jnp.argmin(distances, axis=1)
    avg_distance = jnp.mean(jnp.min(distances, axis=1))
    return cluster_idx, avg_distance

# One K-Means training step.
# @jax.jit compiles the function with XLA, so it runs much faster.
@jax.jit
def kmeans_step(centroids, x):
    # Assign each image to its closest centroid.
    cluster_idx, avg_distance = assign_clusters(centroids, x)
    # Move each centroid to the mean of the images assigned to it.
    # one_hot[i, j] is 1 if image i is assigned to centroid j, 0 otherwise.
    one_hot = jax.nn.one_hot(cluster_idx, k)
    # With the cosine distance, only the direction of an image matters, not
    # its length: normalize the images, so that every image weighs the same.
    x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    sums = jnp.matmul(one_hot.T, x)  # Sum of the images of each cluster.
    counts = jnp.sum(one_hot, axis=0)  # Number of images of each cluster.
    # Keep the old centroid if no image was assigned to it.
    new_centroids = jnp.where(counts[:, None] > 0, sums / jnp.maximum(counts[:, None], 1.),
                              centroids)
    return new_centroids, avg_distance

# %%
# Training. Every step uses the whole training set.
for i in range(1, num_steps + 1):
    centroids, d = kmeans_step(centroids, full_data_x)
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# %%
# Assign a label to each centroid.
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx').
idx, _ = assign_clusters(centroids, full_data_x)
idx = np.asarray(idx)
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i], y_train[i]] += 1
# Assign the most frequent label to the centroid.
labels_map = np.argmax(counts, axis=1)

# Evaluation.
# Assign each test image to its closest centroid.
test_idx, _ = assign_clusters(centroids, x_test)
# Lookup: centroid_id -> label.
cluster_label = labels_map[np.asarray(test_idx)]
# Compute accuracy.
accuracy = np.mean(cluster_label == y_test)

# Test Model.
print("Test Accuracy:", accuracy)
