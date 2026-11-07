""" Neural Network Example.

Build a 2-hidden layers fully connected neural network (a.k.a multilayer
perceptron) with JAX.

This example is using a low-level approach to better understand all
mechanics behind building neural networks and the training process: the
model is a plain Python function, its weights are stored in a dictionary
of arrays, and the gradients are computed with `jax.grad`.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## Neural Network Overview

<img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg" alt="nn" style="width: 400px;"/>

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
num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).

# Training parameters.
# The loss is averaged over the batch, so that it doesn't depend on the batch
# size. Its gradients are smaller than those of a loss summed over the batch,
# so it needs a larger learning rate.
learning_rate = 0.5
training_steps = 3000
batch_size = 256
display_step = 100

# Network parameters.
n_hidden_1 = 128  # 1st layer number of neurons.
n_hidden_2 = 256  # 2nd layer number of neurons.

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
# Store layers weight & bias.

# Random values are generated from a random 'key', which is split to get a
# new key for every use.
key = jax.random.key(0)
key_h1, key_h2, key_out = jax.random.split(key, 3)

# A random value generator to initialize weights.
def random_normal(key, shape):
    return 0.05 * jax.random.normal(key, shape)

# All the model parameters, gathered in a dictionary.
params = {
    'weights': {
        'h1': random_normal(key_h1, [num_features, n_hidden_1]),
        'h2': random_normal(key_h2, [n_hidden_1, n_hidden_2]),
        'out': random_normal(key_out, [n_hidden_2, num_classes]),
    },
    'biases': {
        'b1': jnp.zeros([n_hidden_1]),
        'b2': jnp.zeros([n_hidden_2]),
        'out': jnp.zeros([num_classes]),
    },
}

# %%
# Create model.
def neural_net(params, x):
    weights, biases = params['weights'], params['biases']
    # Hidden fully connected layer with 128 neurons.
    layer_1 = jnp.add(jnp.matmul(x, weights['h1']), biases['b1'])
    # Apply sigmoid to layer_1 output for non-linearity.
    layer_1 = jax.nn.sigmoid(layer_1)

    # Hidden fully connected layer with 256 neurons.
    layer_2 = jnp.add(jnp.matmul(layer_1, weights['h2']), biases['b2'])
    # Apply sigmoid to layer_2 output for non-linearity.
    layer_2 = jax.nn.sigmoid(layer_2)

    # Output fully connected layer with a neuron for each class.
    out_layer = jnp.matmul(layer_2, weights['out']) + biases['out']
    # Apply softmax to normalize the logits to a probability distribution.
    return jax.nn.softmax(out_layer)

# %%
# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = jax.nn.one_hot(y_true, num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = jnp.clip(y_pred, 1e-9, 1.)
    # Compute cross-entropy, averaged across the batch.
    return jnp.mean(-jnp.sum(y_true * jnp.log(y_pred), axis=1))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = jnp.equal(jnp.argmax(y_pred, 1), y_true)
    return jnp.mean(correct_prediction.astype(jnp.float32))

# Stochastic gradient descent optimizer.
optimizer = optax.sgd(learning_rate)
# The optimizer state (e.g. momentum) is stored outside of the optimizer.
opt_state = optimizer.init(params)

# %%
# Optimization process.
# @jax.jit compiles the function with XLA, so it runs much faster
# (see '1_Introduction/transformations.py').
@jax.jit
def run_optimization(params, opt_state, x, y):
    # Define the loss as a function of the parameters, to differentiate it.
    def loss_fn(params):
        pred = neural_net(params, x)
        return cross_entropy(pred, y)

    # Compute gradients: 'jax.grad(loss_fn)' is a new function returning the
    # gradient of the loss with respect to every array in 'params'.
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
        pred = neural_net(params, batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on validation set.
pred = neural_net(params, x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %%
# Visualize predictions.
import matplotlib.pyplot as plt

# %%
# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(params, test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions[i]))
