""" Logistic Regression Example.

Logistic regression implementation with JAX library.

This example is using a low-level approach to better understand all
mechanics behind the training process: the gradients are computed with
`jax.grad`, and the weights are updated with an Optax optimizer.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
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
import optax

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_mnist, batch_iterator

# %%
# MNIST dataset parameters.
num_classes = 10  # 0 to 9 digits
num_features = 784  # 28*28

# Training parameters.
# 0.1 trains faster than a smaller learning rate, such as 0.01: 0.91 test
# accuracy after 1000 steps, instead of 0.87.
learning_rate = 0.1
training_steps = 1000
batch_size = 256
display_step = 50

# %%
# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = load_mnist()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# %%
# Shuffle and batch data (a simple data loader, see 'common/datasets.py').
train_data = batch_iterator(x_train, y_train, batch_size)

# %%
# Weight and Bias, gathered in a dictionary.
params = {
    # Weight of shape [784, 10], the 28*28 image features, and total number of classes.
    'W': jnp.ones([num_features, num_classes]),
    # Bias of shape [10], the total number of classes.
    'b': jnp.zeros([num_classes]),
}

# Logistic regression (Wx + b).
def logistic_regression(params, x):
    # Apply softmax to normalize the logits to a probability distribution.
    return jax.nn.softmax(jnp.matmul(x, params['W']) + params['b'])

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = jax.nn.one_hot(y_true, num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = jnp.clip(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return jnp.mean(-jnp.sum(y_true * jnp.log(y_pred), axis=1))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = jnp.equal(jnp.argmax(y_pred, 1), y_true)
    return jnp.mean(correct_prediction.astype(jnp.float32))

# Stochastic gradient descent optimizer.
optimizer = optax.sgd(learning_rate)
# The optimizer state is stored outside of the optimizer.
opt_state = optimizer.init(params)

# %%
# Optimization process.
# @jax.jit compiles the function with XLA, so it runs much faster
# (see '1_Introduction/transformations.py').
@jax.jit
def run_optimization(params, opt_state, x, y):
    # Define the loss as a function of the parameters, to differentiate it.
    def loss_fn(params):
        pred = logistic_regression(params, x)
        return cross_entropy(pred, y)

    # Compute gradients.
    gradients = jax.grad(loss_fn)(params)

    # Update W and b following gradients.
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)

    # JAX arrays can't be modified in place: return the updated values.
    return params, opt_state

# %%
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Get the next batch of data.
    batch_x, batch_y = next(train_data)
    # Run the optimization to update W and b values.
    params, opt_state = run_optimization(params, opt_state, batch_x, batch_y)

    if step % display_step == 0:
        pred = logistic_regression(params, batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on validation set.
pred = logistic_regression(params, x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %%
# Visualize predictions.
import matplotlib.pyplot as plt

# %%
# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = logistic_regression(params, test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions[i]))
