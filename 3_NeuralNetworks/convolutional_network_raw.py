""" Convolutional Neural Network Example.

Build a convolutional neural network with JAX.

This example is using a low-level approach to better understand all
mechanics behind building convolutional neural networks and the training
process: convolution and pooling are applied with the `jax.lax`
primitives, and the weights are stored in a dictionary of arrays.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## CNN Overview

![CNN](http://personal.ie.cuhk.edu.hk/~ccloy/project_target_code/images/fig3.png)

## MNIST Dataset Overview

This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255.

In this example, each image will be converted to float32 and normalized to [0, 1].

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

# Training parameters.
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# Network parameters.
conv1_filters = 32  # number of filters for 1st conv layer.
conv2_filters = 64  # number of filters for 2nd conv layer.
fc1_units = 1024  # number of neurons for 1st fully-connected layer.

# %%
# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = load_mnist()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# %%
# Shuffle and batch data (a simple data loader, see 'common/datasets.py').
train_data = batch_iterator(x_train, y_train, batch_size)

# %%
# Create some wrappers for simplicity.
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation.
    # 'dimension_numbers' gives the layout of the input (batch, height, width,
    # channels), of the filters (height, width, in channels, out channels)
    # and of the output.
    x = jax.lax.conv_general_dilated(x, W, window_strides=[strides, strides], padding='SAME',
                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    x = x + b
    return jax.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper: take the max over every k x k window.
    # (-inf is the starting value of the max, window size is 1 on the batch
    # and channels dimensions).
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window_dimensions=[1, k, k, 1],
                                 window_strides=[1, k, k, 1], padding='SAME')

# %%
# Store layers weight & bias.
# Random values are generated from a random 'key', which is split to get a
# new key for every use.
keys = jax.random.split(jax.random.key(0), 4)

# A random value generator to initialize weights.
def random_normal(key, shape):
    return 0.05 * jax.random.normal(key, shape)

# All the model parameters, gathered in a dictionary.
params = {
    'weights': {
        # Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
        'wc1': random_normal(keys[0], [5, 5, 1, conv1_filters]),
        # Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
        'wc2': random_normal(keys[1], [5, 5, conv1_filters, conv2_filters]),
        # FC Layer 1: 7*7*64 inputs, 1024 units.
        'wd1': random_normal(keys[2], [7 * 7 * 64, fc1_units]),
        # FC Out Layer: 1024 inputs, 10 units (total number of classes).
        'out': random_normal(keys[3], [fc1_units, num_classes]),
    },
    'biases': {
        'bc1': jnp.zeros([conv1_filters]),
        'bc2': jnp.zeros([conv2_filters]),
        'bd1': jnp.zeros([fc1_units]),
        'out': jnp.zeros([num_classes]),
    },
}

# %%
# Create model.
def conv_net(params, x):
    weights, biases = params['weights'], params['biases']

    # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
    x = jnp.reshape(x, [-1, 28, 28, 1])

    # Convolution Layer. Output shape: [-1, 28, 28, 32].
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer. Output shape: [-1, 14, 14, 64].
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
    conv2 = maxpool2d(conv2, k=2)

    # Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
    fc1 = jnp.reshape(conv2, [-1, weights['wd1'].shape[0]])

    # Fully connected layer, Output shape: [-1, 1024].
    fc1 = jnp.add(jnp.matmul(fc1, weights['wd1']), biases['bd1'])
    # Apply ReLU to fc1 output for non-linearity.
    fc1 = jax.nn.relu(fc1)

    # Fully connected layer, Output shape: [-1, 10].
    out = jnp.add(jnp.matmul(fc1, weights['out']), biases['out'])
    # Apply softmax to normalize the logits to a probability distribution.
    return jax.nn.softmax(out)

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

# ADAM optimizer.
optimizer = optax.adam(learning_rate)
# The optimizer state (Adam's moving averages) is stored outside of the optimizer.
opt_state = optimizer.init(params)

# %%
# Optimization process.
# @jax.jit compiles the function with XLA, so it runs much faster.
@jax.jit
def run_optimization(params, opt_state, x, y):
    # Define the loss as a function of the parameters, to differentiate it.
    def loss_fn(params):
        pred = conv_net(params, x)
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
        pred = conv_net(params, batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on validation set.
pred = conv_net(params, x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %%
# Visualize predictions.
import matplotlib.pyplot as plt

# %%
# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = conv_net(params, test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions[i]))
