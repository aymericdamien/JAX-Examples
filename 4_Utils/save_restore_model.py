""" Save & Restore a Model Example.

Save and Restore a model using JAX. In this example, we will go over both
low and high-level approaches:
- Low-level: save & restore the model parameters (a pytree of arrays).
- High-level: save & restore a Flax model.

Checkpoints are written with Orbax, the checkpointing library of the JAX
ecosystem: it saves any pytree (nested dicts/lists of arrays) to a
directory, and restores it.

This example is using the MNIST database of handwritten digits as toy
dataset (http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_mnist, batch_iterator

# %%
# Directory to save the checkpoints into.
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
# Orbax requires absolute paths.
ckpt_dir = os.path.abspath(os.path.join(out_dir, "jax-example"))

# %%
# MNIST dataset parameters.
num_classes = 10  # 0 to 9 digits
num_features = 784  # 28*28

# Training parameters.
learning_rate = 0.01
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

"""
## 1) Low-level: save & restore a pytree of arrays

Basic logistic regression, with its parameters stored in a dictionary.
"""

# Random values are generated from a random 'key'.
key = jax.random.key(0)
# The model parameters, gathered in a dictionary.
params = {
    # Weight of shape [784, 10], the 28*28 image features, and total number of classes.
    'W': jax.random.normal(key, [num_features, num_classes]),
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
    # Compute cross-entropy, averaged across the batch.
    return jnp.mean(-jnp.sum(y_true * jnp.log(y_pred), axis=1))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = jnp.equal(jnp.argmax(y_pred, 1), y_true)
    return jnp.mean(correct_prediction.astype(jnp.float32))

# Adam optimizer.
optimizer = optax.adam(learning_rate)
# The optimizer state (Adam moving averages) is stored outside of the optimizer.
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

"""
## Save and Load with Orbax
"""

# Save weights and optimizer state.
# Gather everything to save in a single pytree (a dict). Saving the optimizer
# state too lets training resume exactly where it stopped.
vars_to_save = {"params": params, "opt_state": opt_state}
# Orbax checkpointer, to save and restore pytrees of arrays.
checkpointer = ocp.StandardCheckpointer()
# Path to the checkpoint.
ckpt_path = os.path.join(ckpt_dir, "logistic_regression")

# %%
# Save variables ('force=True' overwrites the checkpoint of a previous run).
checkpointer.save(ckpt_path, vars_to_save, force=True)
# Saving runs in the background: wait until it is done.
checkpointer.wait_until_finished()

# %%
# Check weight value, and model accuracy.
print("W mean: %f" % jnp.mean(params['W']))
print("Test Accuracy: %f" % accuracy(logistic_regression(params, x_test), y_test))

# %%
# Reset variables to test restore.
params = {
    'W': jax.random.normal(jax.random.key(1), [num_features, num_classes]),
    'b': jnp.zeros([num_classes]),
}
opt_state = optimizer.init(params)

# %%
# Check resetted weight value, and model accuracy.
print("W mean: %f" % jnp.mean(params['W']))
print("Test Accuracy: %f" % accuracy(logistic_regression(params, x_test), y_test))

# %%
# Restore variables from the checkpoint.
# The current variables are given as a template: they tell Orbax the
# structure, shapes and dtypes of the arrays to restore.
restored = checkpointer.restore(ckpt_path, {"params": params, "opt_state": opt_state})
params, opt_state = restored["params"], restored["opt_state"]

# %%
# Confirm that W has been correctly restored (identical accuracy).
print("W mean: %f" % jnp.mean(params['W']))
print("Test Accuracy: %f" % accuracy(logistic_regression(params, x_test), y_test))

"""
## 2) High-level: Flax Model

Basic neural network with a Flax model.
"""

# Training parameters.
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 100

# %%
# Create Flax Model.
class NeuralNet(nnx.Module):
    # Set layers.
    # 'rngs' provides the random keys used to initialize the layers weights.
    def __init__(self, rngs):
        # First fully-connected hidden layer.
        self.fc1 = nnx.Linear(num_features, 64, rngs=rngs)
        # Second fully-connected hidden layer.
        self.fc2 = nnx.Linear(64, 128, rngs=rngs)
        # Output fully-connected layer, with a neuron for each class.
        self.out = nnx.Linear(128, num_classes, rngs=rngs)

    # Set forward pass.
    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        # Return the raw scores (logits): the loss function applies softmax.
        return self.out(x)

# Build neural network model.
neural_net = NeuralNet(rngs=nnx.Rngs(0))

# %%
# Cross-Entropy loss function.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=x, labels=y)
    return jnp.mean(loss)

# Adam optimizer.
optimizer = nnx.Optimizer(neural_net, optax.adam(learning_rate), wrt=nnx.Param)

# %%
# Optimization process.
@nnx.jit
def run_optimization(neural_net, optimizer, x, y):
    # Define the loss as a function of the model, to differentiate it.
    def loss_fn(neural_net):
        pred = neural_net(x)
        return cross_entropy_loss(pred, y)

    # Compute gradients.
    gradients = nnx.grad(loss_fn)(neural_net)

    # Update the model parameters following gradients.
    optimizer.update(neural_net, gradients)

# %%
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Get the next batch of data.
    batch_x, batch_y = next(train_data)
    # Run the optimization to update the model parameters.
    run_optimization(neural_net, optimizer, batch_x, batch_y)

    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

"""
## Save and Load with Flax Model
"""

# Save Flax model.
# 'nnx.state' extracts all the model variables (weights & biases) as a
# pytree of arrays, that Orbax can save.
model_path = os.path.join(ckpt_dir, "neural_net")
checkpointer.save(model_path, nnx.state(neural_net), force=True)
checkpointer.wait_until_finished()

# %%
# Test model performance.
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %%
# Re-build neural network model with default values (new random weights).
neural_net = NeuralNet(rngs=nnx.Rngs(1))
# Test model performance.
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %%
# Load saved weights.
# The state of the new model is the template to restore into.
state = checkpointer.restore(model_path, nnx.state(neural_net))
# Copy the restored weights into the model.
nnx.update(neural_net, state)

# %%
# Test that weights loaded correctly.
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
