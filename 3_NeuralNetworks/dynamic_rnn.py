""" Dynamic Recurrent Neural Network Example.

JAX implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length, using Flax
'layers' and 'model' API. This example is using a toy dataset to classify
linear sequences. The generated sequences have variable length.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## RNN Overview

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" alt="nn" style="width: 600px;"/>

References:
- [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), Sepp Hochreiter & Jurgen Schmidhuber, Neural Computation 9(8): 1735-1780, 1997.
"""

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# %%
# Dataset parameters.
num_classes = 2  # linear sequence or not.
seq_max_len = 20  # Maximum sequence length.
seq_min_len = 5  # Minimum sequence length (before padding).
masking_val = -1  # -1 will represents the mask and be used to pad sequences to a common max length.
max_value = 10000  # Maximum int value.

# Training Parameters.
learning_rate = 0.001
training_steps = 2000
batch_size = 64
display_step = 100

# Network Parameters.
num_units = 32  # number of neurons for the LSTM layer.

# %%
# ====================
#  TOY DATA GENERATOR
# ====================

# A random number generator (with a fixed seed, to get the same data at every run).
rng = np.random.default_rng(0)

def toy_sequence_data():
    """ Generate sequence of data with dynamic length.
    This function generates toy samples for training:
    - Class 0: linear sequences (i.e. [1, 2, 3, 4, ...])
    - Class 1: random sequences (i.e. [9, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'seq_max_len' for dimension
    consistency (all the sequences of a batch are stored in a single array).
    The dynamic calculation will then be performed and ignore the masked
    value (here -1).
    """
    while True:
        # Set variable sequence length.
        seq_len = rng.integers(seq_min_len, seq_max_len + 1)
        rand_start = rng.integers(0, max_value - seq_len + 1)
        # Add a random or linear int sequence (50% prob).
        if rng.random() < .5:
            # Generate a linear sequence.
            seq = np.arange(start=rand_start, stop=rand_start + seq_len)
            # Rescale values to [0., 1.].
            seq = seq / max_value
            # Pad sequence until the maximum length for dimension consistency.
            # Masking value: -1.
            seq = np.pad(seq, mode='constant', pad_width=(0, seq_max_len - seq_len), constant_values=masking_val)
            label = 0
        else:
            # Generate a random sequence.
            seq = rng.integers(max_value, size=seq_len)
            # Rescale values to [0., 1.].
            seq = seq / max_value
            # Pad sequence until the maximum length for dimension consistency.
            # Masking value: -1.
            seq = np.pad(seq, mode='constant', pad_width=(0, seq_max_len - seq_len), constant_values=masking_val)
            label = 1
        yield np.array(seq, dtype=np.float32), np.array(label, dtype=np.int32)

# %%
# Batch data: stack 'batch_size' samples from the data generator.
def next_batch(data_generator, batch_size):
    samples = [next(data_generator) for _ in range(batch_size)]
    batch_x = np.stack([seq for seq, label in samples])
    batch_y = np.stack([label for seq, label in samples])
    return batch_x, batch_y

train_data = toy_sequence_data()

# %%
# Create LSTM Model.
class LSTM(nnx.Module):
    # Set layers.
    # 'rngs' provides the random keys used to initialize the layers weights.
    def __init__(self, rngs):
        # Define a LSTM layer: 'nnx.RNN' runs the LSTM cell over every
        # timestep. Each timestep has a single value (1 input feature).
        self.lstm = nnx.RNN(nnx.LSTMCell(1, num_units, rngs=rngs))
        # Output fully connected layer (2 classes: linear or random seq).
        self.out = nnx.Linear(num_units, num_classes, rngs=rngs)

    # Set forward pass.
    def __call__(self, x):
        # A RNN Layer expects a 3-dim input (batch_size, seq_len, num_features).
        x = jnp.reshape(x, [-1, seq_max_len, 1])
        # Masking: padded timesteps hold 'masking_val', so the true length
        # of each sequence is its number of non-masked values.
        seq_lengths = jnp.sum(x[:, :, 0] != masking_val, axis=1)
        # Apply LSTM layer. Given 'seq_lengths', the returned LSTM state (with
        # 'return_carry=True') is the state at the last real timestep of each
        # sequence: the padded timesteps are ignored, and 'h' is the LSTM
        # output at each sequence's true last step.
        (c, h), outputs = self.lstm(x, seq_lengths=seq_lengths, return_carry=True)
        # Apply output layer.
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
    batch_x, batch_y = next_batch(train_data, batch_size)
    # Run the optimization to update the model parameters.
    run_optimization(lstm_net, optimizer, batch_x, batch_y)

    if step % display_step == 0 or step == 1:
        pred = lstm_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on 1000 new sequences.
test_x, test_y = next_batch(toy_sequence_data(), 1000)
pred = lstm_net(test_x)
print("Test Accuracy: %f" % accuracy(pred, test_y))
