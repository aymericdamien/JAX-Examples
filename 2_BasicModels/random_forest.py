""" Random Forest Example.

Implement Random Forest algorithm with JAX, and apply it to classify
handwritten digit images. This example is using the MNIST database of
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).

The forest is built from scratch: JAX scores all the candidate splits of a
tree node at once (`jax.jit`), and runs the trees on many images in
parallel (`jax.vmap`).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## Random Forest Overview

A random forest is a set of decision trees, each one trained on a slightly different version of the data, that vote to classify a sample. Two sources of randomness make the trees different from each other:

- **Bagging:** every tree is trained on a 'bootstrap' sample of the training set (samples drawn at random, with replacement).
- **Random features:** to split a node, a tree only considers a random subset of the features.

Each tree is grown greedily, from the root: a node is split in two with the test `pixel[feature] <= threshold` that best separates the classes (measured by the Gini impurity), until the nodes are pure, too small or too deep. A leaf predicts the class distribution of its training samples, and the forest averages the predictions of all its trees.

## MNIST Dataset Overview

This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255.

In this example, each image will be converted to float32, normalized to [0, 1] and flattened to a 1-D array of 784 features (28*28).

![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)

More info: http://yann.lecun.com/exdb/mnist/
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
# MNIST dataset parameters.
num_classes = 10  # The 10 digits.
num_features = 784  # Each image is 28x28 pixels.

# Training parameters.
num_samples = 20000  # Train on a subset of MNIST, to keep the example fast.

# Random Forest parameters.
num_trees = 10
max_depth = 12  # Maximum depth of a tree.
min_samples_leaf = 5  # Minimum number of samples in a leaf.
num_split_features = 28  # Features considered for each split (~sqrt(784)).
# Candidate thresholds of a split, on the [0, 1] pixel values.
thresholds = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

# %%
# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = load_mnist()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.
# Keep the first 'num_samples' images for training.
x_train, y_train = x_train[:num_samples], y_train[:num_samples]

# %%
# Trees are grown by plain Python code, so we use NumPy's random generator
# for the random choices (bootstrap samples and features of each split).
rng = np.random.default_rng(0)

"""
### Build the Random Forest

A tree is stored in a few arrays (one entry per node): the `feature` and `threshold` used to split the node (samples with `x[feature] <= threshold` go to the `left` child, the others to the `right` child), and the class distribution `value` predicted by the node when it is a leaf.
"""

# Gini impurity of a node, given the number of samples of each class:
# the probability that two samples drawn at random have different classes.
# 0 means the node is pure (it only contains one class).
def gini(class_counts):
    num = jnp.sum(class_counts, axis=-1)
    prob = class_counts / jnp.maximum(num, 1)[..., None]
    return 1. - jnp.sum(prob ** 2, axis=-1)

# Find the best split of a node, among the given features and all the
# candidate thresholds. The node is given as a mask over all the training
# samples, so the arrays always have the same shape and the function is
# only compiled once.
@jax.jit
def find_best_split(x, labels, node_mask, features):
    # One-hot labels of the node samples (other samples are all zeros).
    node_labels = jax.nn.one_hot(labels, num_classes) * node_mask[:, None]

    # goes_left[s, f, t] is 1 if sample 's' goes to the left child when
    # splitting on features[f] with thresholds[t].
    # Shape: (num_samples, num_split_features, num_thresholds).
    goes_left = (x[:, features][:, :, None] <= thresholds).astype(jnp.float32)

    # Number of samples of each class in the left and right children, for
    # every candidate split ('einsum' sums over the samples 's').
    # Shape: (num_split_features, num_thresholds, num_classes).
    left_counts = jnp.einsum('sft,sc->ftc', goes_left, node_labels)
    right_counts = jnp.sum(node_labels, axis=0) - left_counts

    # Impurity after the split: the children impurities, weighted by size.
    num_left = jnp.sum(left_counts, axis=-1)
    num_right = jnp.sum(right_counts, axis=-1)
    impurity = ((num_left * gini(left_counts) + num_right * gini(right_counts))
                / (num_left + num_right))
    # A split is only valid if both children have enough samples.
    valid = (num_left >= min_samples_leaf) & (num_right >= min_samples_leaf)
    impurity = jnp.where(valid, impurity, jnp.inf)

    # Pick the split with the lowest impurity.
    best_feature, best_threshold = jnp.unravel_index(jnp.argmin(impurity), impurity.shape)
    return (features[best_feature], thresholds[best_threshold],
            impurity[best_feature, best_threshold])

# %%
# Grow a tree, recursively, from its training samples.
def grow_tree(x, labels):
    # Copy the data to the device once, instead of at every node.
    x, labels = jnp.asarray(x), jnp.asarray(labels)
    tree = {'feature': [], 'threshold': [], 'left': [], 'right': [], 'value': []}

    def add_node(node_mask, depth):
        node_id = len(tree['feature'])
        # Start as a leaf. A leaf points to itself (left = right = node_id),
        # so walking down the tree stops there.
        tree['feature'].append(0)
        tree['threshold'].append(0.)
        tree['left'].append(node_id)
        tree['right'].append(node_id)
        # Leaf value: the class distribution of the node samples.
        class_counts = jnp.sum(jax.nn.one_hot(labels, num_classes) * node_mask[:, None], axis=0)
        num_node_samples = jnp.sum(class_counts)
        tree['value'].append(class_counts / num_node_samples)

        # Split the node, unless it is too deep, too small or already pure.
        node_impurity = gini(class_counts)
        if (depth < max_depth and num_node_samples >= 2 * min_samples_leaf
                and node_impurity > 0):
            # Only consider a random subset of the features for this split.
            features = rng.choice(num_features, num_split_features, replace=False)
            feature, threshold, impurity = find_best_split(x, labels, node_mask, features)
            # Only split if it makes the children purer than the parent.
            if impurity < node_impurity:
                goes_left = x[:, feature] <= threshold
                tree['feature'][node_id] = feature
                tree['threshold'][node_id] = threshold
                tree['left'][node_id] = add_node(node_mask & goes_left, depth + 1)
                tree['right'][node_id] = add_node(node_mask & ~goes_left, depth + 1)
        return node_id

    # The root node contains all the samples.
    add_node(jnp.ones(len(x), dtype=bool), depth=0)
    return {k: jnp.array(v) for k, v in tree.items()}

# %%
# Predict the class distribution of a single image 'x' with a tree, by
# walking down from the root: 'max_depth' moves always reach a leaf (leaves
# point to themselves).
def predict_tree(tree, x):
    node = 0
    for _ in range(max_depth):
        go_left = x[tree['feature'][node]] <= tree['threshold'][node]
        node = jnp.where(go_left, tree['left'][node], tree['right'][node])
    return tree['value'][node]

# 'predict_tree' handles a single image: jax.vmap turns it into a function
# predicting a whole batch of images at once (in_axes: same tree, batch of x).
predict_tree_batch = jax.vmap(predict_tree, in_axes=(None, 0))

# The forest prediction is the average of all the trees predictions.
def predict_forest(forest, x):
    return jnp.mean(jnp.stack([predict_tree_batch(tree, x) for tree in forest]), axis=0)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = jnp.equal(jnp.argmax(y_pred, 1), y_true)
    return jnp.mean(correct_prediction.astype(jnp.float32))

# %%
# Training: grow the trees one after the other.
forest = []
for i in range(1, num_trees + 1):
    # Bagging: train the tree on a bootstrap sample of the training set
    # (samples drawn at random, with replacement).
    bootstrap = rng.integers(0, num_samples, num_samples)
    tree = grow_tree(x_train[bootstrap], y_train[bootstrap])
    forest.append(tree)

    # Accuracy of the forest built so far, on the training set.
    acc = accuracy(predict_forest(forest, x_train), y_train)
    print("tree: %i, nodes: %i, accuracy: %f" % (i, len(tree['feature']), acc))

# Test model on validation set.
pred = predict_forest(forest, x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %%
# Visualize predictions.
import matplotlib.pyplot as plt

# %%
# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = predict_forest(forest, test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions[i]))
