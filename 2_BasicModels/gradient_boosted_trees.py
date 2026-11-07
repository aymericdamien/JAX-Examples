""" Gradient Boosted Decision Tree (GBDT) Example.

Implement a Gradient Boosted Decision Tree (GBDT) with JAX. This example is
using the Boston Housing Value dataset as training samples. The example
supports both Classification (2 classes: value > $23000 or not) and
Regression (raw home value as target).

The GBDT is built from scratch: JAX computes the gradients of the loss
(`jax.grad`), scores all the candidate splits of a tree node at once
(`jax.jit`), and runs the trees on many samples in parallel (`jax.vmap`).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## Boston Housing Dataset

**Link:** https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

**Description:**

The dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been used extensively throughout the literature to benchmark algorithms. However, these comparisons were primarily done outside of Delve and are thus somewhat suspect. The dataset is small in size with only 506 cases.

The data was originally published by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.`

*For the full features list, please see the link above*

## Gradient Boosting Overview

A GBDT is a sum of small decision trees. The trees are trained one after the other: each new tree is fitted to correct the errors of the trees built so far.

More precisely, for every training sample, we compute the gradient `g` (and the second derivative `h`) of the loss with respect to the current prediction. A new tree is grown to group samples with similar gradients in the same leaf, and each leaf predicts the step `-sum(g) / (sum(h) + l2_regul)` (a Newton step). The new tree output is then added to the model prediction, scaled by the learning rate.

Because only `g` and `h` depend on the loss function, the same algorithm handles both classification (logistic loss) and regression (squared error loss), and JAX computes `g` and `h` for us with `jax.grad`.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_boston_housing

# %%
# Dataset parameters.
num_classes = 2  # Total classes: greater or equal to $23,000, or not (See notes below).
num_features = 13  # data features size.

# Training parameters.
learning_rate = 1.0
l2_regul = 0.1
# Minimum sum of hessians in each child of a split.
# Without it, the trees can isolate single samples, which overfits the
# training set (here, test accuracy drops from 0.85 to 0.82).
min_node_weight = 1.0

# GBDT parameters.
num_trees = 10
max_depth = 4

# %%
# Prepare Boston Housing Dataset.
(x_train, y_train), (x_test, y_test) = load_boston_housing()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
y_train, y_test = np.array(y_train, np.float32), np.array(y_test, np.float32)

# For classification purpose, we build 2 classes: price greater or lower than $23,000.
def to_binary_class(y):
    return np.array(y >= 23.0, np.float32)

y_train_binary = to_binary_class(y_train)
y_test_binary = to_binary_class(y_test)

"""
### Build the GBDT

A tree is stored in a few arrays (one entry per node): the `feature` and `threshold` used to split the node (samples with `x[feature] <= threshold` go to the `left` child, the others to the `right` child), and the `value` predicted by the node when it is a leaf.
"""

# Score of a node, where G and H are the sums of the gradients and hessians
# of its samples: a leaf predicting -G / (H + l2_regul) decreases the loss by
# about half this score. A split is good if the two children score more than
# the parent.
def node_score(g_sum, h_sum):
    return g_sum ** 2 / (h_sum + l2_regul)

# Find the best split of a node. Every value of every feature (in the
# training set) is a candidate threshold, and all of them are scored at once.
# The node is given as a mask over all the training samples, so the arrays
# always have the same shape and the function is only compiled once.
@jax.jit
def find_best_split(x, g, h, node_mask):
    # Ignore the samples that are not in this node.
    g = jnp.where(node_mask, g, 0.)
    h = jnp.where(node_mask, h, 0.)
    count = node_mask.astype(jnp.float32)

    # goes_left[t, s, f] is True if sample 's' goes to the left child when
    # splitting on feature 'f' with threshold x[t, f].
    # Shape: (num_thresholds, num_samples, num_features).
    goes_left = x[None, :, :] <= x[:, None, :]

    # Sums over the samples of the left child, for every candidate split.
    # Shape: (num_thresholds, num_features).
    g_left = jnp.sum(goes_left * g[None, :, None], axis=1)
    h_left = jnp.sum(goes_left * h[None, :, None], axis=1)
    count_left = jnp.sum(goes_left * count[None, :, None], axis=1)
    # The right child gets the rest of the node samples.
    g_right = jnp.sum(g) - g_left
    h_right = jnp.sum(h) - h_left
    count_right = jnp.sum(count) - count_left

    # Gain: score of the two children minus the score of the parent.
    gain = (node_score(g_left, h_left) + node_score(g_right, h_right)
            - node_score(jnp.sum(g), jnp.sum(h)))
    # A split is only valid if both children have at least one sample, and
    # enough hessian weight.
    valid = ((count_left > 0) & (count_right > 0)
             & (h_left >= min_node_weight) & (h_right >= min_node_weight))
    gain = jnp.where(valid, gain, -jnp.inf)

    # Pick the candidate with the highest gain.
    best_threshold, best_feature = jnp.unravel_index(jnp.argmax(gain), gain.shape)
    return best_feature, x[best_threshold, best_feature], gain[best_threshold, best_feature]

# %%
# Grow a tree, recursively, from the gradients and hessians of the samples.
def grow_tree(x, g, h):
    # Copy the data to the device once, instead of at every node.
    x = jnp.asarray(x)
    tree = {'feature': [], 'threshold': [], 'left': [], 'right': [], 'value': []}

    def add_node(node_mask, depth):
        node_id = len(tree['feature'])
        # Start as a leaf. A leaf points to itself (left = right = node_id),
        # so walking down the tree stops there.
        tree['feature'].append(0)
        tree['threshold'].append(0.)
        tree['left'].append(node_id)
        tree['right'].append(node_id)
        # Leaf value: the Newton step -G / (H + l2_regul).
        g_sum, h_sum = jnp.sum(g * node_mask), jnp.sum(h * node_mask)
        tree['value'].append(-g_sum / (h_sum + l2_regul))

        # Split the node, unless the tree is deep enough or no split helps.
        if depth < max_depth:
            feature, threshold, gain = find_best_split(x, g, h, node_mask)
            if gain > 0:
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
# Predict the value of a single sample 'x' with a tree, by walking down from
# the root: 'max_depth' moves always reach a leaf (leaves point to themselves).
def predict_tree(tree, x):
    node = 0
    for _ in range(max_depth):
        go_left = x[tree['feature'][node]] <= tree['threshold'][node]
        node = jnp.where(go_left, tree['left'][node], tree['right'][node])
    return tree['value'][node]

# 'predict_tree' handles a single sample: jax.vmap turns it into a function
# predicting a whole batch of samples at once (in_axes: same tree, batch of x).
predict_tree_batch = jax.vmap(predict_tree, in_axes=(None, 0))

# The GBDT prediction is the sum of all the trees outputs.
def predict(trees, x):
    pred = jnp.zeros(len(x))
    for tree in trees:
        pred = pred + learning_rate * predict_tree_batch(tree, x)
    return pred

# %%
# Train a GBDT for the given per-sample loss function 'loss_fn(pred, y)'.
def train_gbdt(x, y, loss_fn):
    # Gradient and hessian (2nd derivative) of the loss with respect to the
    # prediction, computed by JAX automatic differentiation. jax.vmap applies
    # them to every sample of the training set at once.
    grad_fn = jax.vmap(jax.grad(loss_fn))
    hessian_fn = jax.vmap(jax.grad(jax.grad(loss_fn)))

    trees = []
    # The model starts by predicting 0 for every sample.
    pred = jnp.zeros(len(x))
    for i in range(1, num_trees + 1):
        # Fit a new tree to the gradients of the current predictions.
        g, h = grad_fn(pred, y), hessian_fn(pred, y)
        tree = grow_tree(x, g, h)
        trees.append(tree)
        # Add the new tree to the model.
        pred = pred + learning_rate * predict_tree_batch(tree, x)
        loss = jnp.mean(jax.vmap(loss_fn)(pred, y))
        print("tree: %i, loss: %f" % (i, loss))
    return trees

"""
### GBDT Classifier
"""

# Logistic loss: the model predicts a score (logit), and sigmoid(score) is
# the probability of the sample to be of class 1.
def logistic_loss(pred, y):
    return optax.sigmoid_binary_cross_entropy(pred, y)

# Accuracy metric.
def accuracy(pred, y_true):
    # The predicted class is 1 if the probability sigmoid(pred) is > 0.5,
    # i.e. if pred > 0.
    correct_prediction = jnp.equal(pred > 0, y_true == 1)
    return jnp.mean(correct_prediction.astype(jnp.float32))

# %%
# Train the GBDT classifier.
gbdt_classifier = train_gbdt(x_train, y_train_binary, logistic_loss)

# %%
# Evaluate the classifier on the training set.
pred = predict(gbdt_classifier, x_train)
print("Train loss: %f, accuracy: %f" % (
    jnp.mean(logistic_loss(pred, y_train_binary)), accuracy(pred, y_train_binary)))

# %%
# Evaluate the classifier on the test set.
pred = predict(gbdt_classifier, x_test)
print("Test loss: %f, accuracy: %f" % (
    jnp.mean(logistic_loss(pred, y_test_binary)), accuracy(pred, y_test_binary)))

"""
### GBDT Regressor
"""

# Squared error loss. Its gradient 2 * (pred - y) is proportional to the
# error, and its hessian is 2: each new tree is fitted to the remaining
# errors (residuals) of the model.
def squared_loss(pred, y):
    return (pred - y) ** 2

# %%
# Train the GBDT regressor.
gbdt_regressor = train_gbdt(x_train, y_train, squared_loss)

# %%
# Evaluate the regressor on the test set (mean squared error).
pred = predict(gbdt_regressor, x_test)
print("Test loss (MSE): %f" % jnp.mean(squared_loss(pred, y_test)))

# %%
# Compare a few predictions with the true house values (in $1000).
for i in range(5):
    print("Predicted value: %.1f, True value: %.1f" % (pred[i], y_test[i]))
