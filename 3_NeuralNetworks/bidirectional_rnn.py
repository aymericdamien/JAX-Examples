""" Bi-directional Recurrent Neural Network Example.

Build a bi-directional recurrent neural network (LSTM) with JAX, using Flax
'layers' and 'model' API.

A bi-directional RNN runs two LSTMs over the sequence: one reads it from
the first to the last timestep, the other from the last to the first. Flax
(NNX) provides this as the `nnx.Bidirectional` layer. The optimizer comes
from Optax.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## BiRNN Overview

<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/191dd7df9cb91ac22f56ed0dfa4a5651e8767a51/1-Figure2-1.png" alt="nn" style="width: 600px;"/>

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
class BiRNN(nnx.Module):
    # Set layers.
    # 'rngs' provides the random keys used to initialize the layers weights.
    def __init__(self, rngs):
        # Define 2 LSTM layers for forward and backward sequences.
        lstm_fw = nnx.RNN(nnx.LSTMCell(num_input, num_units, rngs=rngs))
        lstm_bw = nnx.RNN(nnx.LSTMCell(num_input, num_units, rngs=rngs))
        # BiRNN layer (it feeds the sequence to 'lstm_bw' in reverse order).
        self.bi_lstm = nnx.Bidirectional(lstm_fw, lstm_bw)
        # Output fully-connected layer (num_classes). It takes the outputs
        # of both LSTMs, concatenated: 2 * num_units values.
        self.out = nnx.Linear(2 * num_units, num_classes, rngs=rngs)

    # Set forward pass.
    def __call__(self, x):
        # BiRNN layer. With 'return_carry=True', it also returns the final
        # state (cell state 'c', hidden state 'h') of each LSTM: 'h_fw' is
        # the forward LSTM output after reading the whole sequence, 'h_bw'
        # is the backward LSTM output after reading it in reverse order.
        ((c_fw, h_fw), (c_bw, h_bw)), outputs = self.bi_lstm(x, return_carry=True)
        x = jnp.concatenate([h_fw, h_bw], axis=-1)
        # Output layer (num_classes).
        # Return the raw scores (logits): the loss function applies softmax.
        return self.out(x)

# Build LSTM model.
birnn_net = BiRNN(rngs=nnx.Rngs(0))

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
optimizer = nnx.Optimizer(birnn_net, optax.adam(learning_rate), wrt=nnx.Param)

# %%
# Optimization process.
# @nnx.jit compiles the function with XLA, so it runs much faster
# (see '1_Introduction/transformations.py').
@nnx.jit
def run_optimization(birnn_net, optimizer, x, y):
    # Define the loss as a function of the model, to differentiate it.
    def loss_fn(birnn_net):
        # Forward pass.
        pred = birnn_net(x)
        # Compute loss.
        return cross_entropy_loss(pred, y)

    # Compute gradients of the loss with respect to the model parameters.
    gradients = nnx.grad(loss_fn)(birnn_net)

    # Update the model parameters following gradients.
    optimizer.update(birnn_net, gradients)

# %%
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Get the next batch of data.
    batch_x, batch_y = next(train_data)
    # Run the optimization to update the model parameters.
    run_optimization(birnn_net, optimizer, batch_x, batch_y)

    if step % display_step == 0:
        pred = birnn_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on validation set.
pred = birnn_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
