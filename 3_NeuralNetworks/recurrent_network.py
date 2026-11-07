""" Recurrent Neural Network Example.

Build a recurrent neural network (LSTM) with JAX, using Flax 'layers' and
'model' API.

Flax (NNX) provides the LSTM cell (`nnx.LSTMCell`) and the layer that runs
a cell over every timestep of a sequence (`nnx.RNN`). The optimizer comes
from Optax.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## RNN Overview

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" alt="nn" style="width: 600px;"/>

References:
- [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), Sepp Hochreiter & Jurgen Schmidhuber, Neural Computation 9(8): 1735-1780, 1997.

## MNIST Dataset Overview

This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255.

In this example, each image will be converted to float32 and normalized to [0, 1].

![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)

To classify images using a recurrent neural network, we consider every image as a sequence of rows: each row of pixels is one timestep. Because MNIST image shape is 28*28px, we will then handle a sequence of 28 timesteps, with 28 values each, for every sample.

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
num_features = 784  # data features (img shape: 28*28).

# Training Parameters.
learning_rate = 0.001
training_steps = 1000
batch_size = 32
display_step = 100

# Network Parameters.
# MNIST image shape is 28*28px: every sample is a sequence of 28 timesteps (the
# rows of the image), with 28 values each (the pixels of a row).
num_input = 28  # number of values per timestep (the pixels of a row).
timesteps = 28  # number of timesteps (the rows of an image).
num_units = 32  # number of neurons for the LSTM layer.

# %%
# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = load_mnist()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Reshape images to sequences of 28 timesteps with 28 values each (28*28).
x_train, x_test = x_train.reshape([-1, timesteps, num_input]), x_test.reshape([-1, timesteps, num_input])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# %%
# Shuffle and batch data (a simple data loader, see 'common/datasets.py').
train_data = batch_iterator(x_train, y_train, batch_size)

# %%
# Create LSTM Model.
class LSTM(nnx.Module):
    # Set layers.
    # 'rngs' provides the random keys used to initialize the layers weights.
    def __init__(self, rngs):
        # RNN (LSTM) hidden layer: 'nnx.RNN' runs the LSTM cell over every timestep.
        self.lstm_layer = nnx.RNN(nnx.LSTMCell(num_input, num_units, rngs=rngs))
        # Output fully-connected layer (num_classes).
        self.out = nnx.Linear(num_units, num_classes, rngs=rngs)

    # Set forward pass.
    def __call__(self, x):
        # LSTM layer. Input shape: [batch_size, timesteps, num_input].
        # With 'return_carry=True', it also returns the final LSTM state
        # (cell state 'c', hidden state 'h'): 'h' is the output of the last timestep.
        (c, h), outputs = self.lstm_layer(x, return_carry=True)
        # Output layer (num_classes), applied to the last LSTM output.
        # Return the raw scores (logits): the loss function applies softmax.
        return self.out(h)

# Build LSTM model.
lstm_net = LSTM(rngs=nnx.Rngs(0))

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
# 'wrt=nnx.Param' tells the optimizer to update all the model parameters.
optimizer = nnx.Optimizer(lstm_net, optax.adam(learning_rate), wrt=nnx.Param)

# %%
# Optimization process.
# @nnx.jit compiles the function with XLA, so it runs much faster
# (see '1_Introduction/transformations.py').
@nnx.jit
def run_optimization(lstm_net, optimizer, x, y):
    # Define the loss as a function of the model, to differentiate it.
    def loss_fn(lstm_net):
        # Forward pass.
        pred = lstm_net(x)
        # Compute loss.
        return cross_entropy_loss(pred, y)

    # Compute gradients of the loss with respect to the model parameters.
    gradients = nnx.grad(loss_fn)(lstm_net)

    # Update the model parameters following gradients.
    optimizer.update(lstm_net, gradients)

# %%
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Get the next batch of data.
    batch_x, batch_y = next(train_data)
    # Run the optimization to update the model parameters.
    run_optimization(lstm_net, optimizer, batch_x, batch_y)

    if step % display_step == 0:
        pred = lstm_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on validation set.
pred = lstm_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
