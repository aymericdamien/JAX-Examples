""" Convolutional Neural Network Example.

Build a convolutional neural network with JAX, using Flax 'layers' and
'model' API (`nnx.Conv`, `nnx.Linear`, `nnx.Dropout`, ...).

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

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

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
# Create Flax Model.
class ConvNet(nnx.Module):
    # Set layers.
    def __init__(self, rngs):
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = nnx.Conv(1, conv1_filters, kernel_size=(5, 5), padding='VALID', rngs=rngs)
        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = nnx.Conv(conv1_filters, conv2_filters, kernel_size=(3, 3), padding='VALID',
                              rngs=rngs)
        # Fully connected layer (its input is the flattened 5x5x64 conv2 output).
        self.fc1 = nnx.Linear(5 * 5 * conv2_filters, fc1_units, rngs=rngs)
        # Apply Dropout (only in training mode, see 'train()' and 'eval()' below).
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
        # Output layer, class prediction.
        self.out = nnx.Linear(fc1_units, num_classes, rngs=rngs)

    # Set forward pass.
    def __call__(self, x):
        # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
        x = jnp.reshape(x, [-1, 28, 28, 1])
        # Convolution Layer. Output shape: [-1, 24, 24, 32].
        x = nnx.relu(self.conv1(x))
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        # Output shape: [-1, 12, 12, 32].
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Convolution Layer. Output shape: [-1, 10, 10, 64].
        x = nnx.relu(self.conv2(x))
        # Max Pooling (down-sampling). Output shape: [-1, 5, 5, 64].
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Flatten the data to a 1-D vector for the fully connected layer.
        x = jnp.reshape(x, [x.shape[0], -1])
        # Fully connected layer. Output shape: [-1, 1024].
        x = self.fc1(x)
        x = self.dropout(x)
        # Output layer. Output shape: [-1, 10] (the logits).
        return self.out(x)

# Build neural network model.
# 'rngs' provides the random keys to initialize the weights and for dropout.
conv_net = ConvNet(rngs=nnx.Rngs(0))

# %%
# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Apply softmax to logits and compute cross-entropy.
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=x, labels=y)
    # Average loss across the batch.
    return jnp.mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = jnp.equal(jnp.argmax(y_pred, 1), y_true)
    return jnp.mean(correct_prediction.astype(jnp.float32))

# Adam optimizer.
optimizer = nnx.Optimizer(conv_net, optax.adam(learning_rate), wrt=nnx.Param)

# %%
# Optimization process.
# @nnx.jit compiles the function with XLA, so it runs much faster.
@nnx.jit
def run_optimization(conv_net, optimizer, x, y):
    # Define the loss as a function of the model, to differentiate it.
    def loss_fn(conv_net):
        # Forward pass.
        pred = conv_net(x)
        # Compute loss.
        return cross_entropy_loss(pred, y)

    # Compute gradients of the loss with respect to the model parameters.
    gradients = nnx.grad(loss_fn)(conv_net)

    # Update the model parameters following gradients.
    optimizer.update(conv_net, gradients)

# %%
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Get the next batch of data.
    batch_x, batch_y = next(train_data)
    # Set the model in training mode (dropout is applied).
    conv_net.train()
    # Run the optimization to update the model parameters.
    run_optimization(conv_net, optimizer, batch_x, batch_y)

    if step % display_step == 0:
        # Set the model in evaluation mode (dropout is not applied).
        conv_net.eval()
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on validation set.
conv_net.eval()
pred = conv_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %%
# Visualize predictions.
import matplotlib.pyplot as plt

# %%
# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = conv_net(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions[i]))
