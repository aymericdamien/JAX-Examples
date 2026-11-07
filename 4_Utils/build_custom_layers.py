""" Build Custom Layers & Modules Example.

Build custom layers and modules with JAX, using Flax.

In Flax, every layer (`nnx.Linear`, `nnx.Conv`, ...) and every model is a
`nnx.Module`: a Python class holding its own weights as `nnx.Param`
attributes. A custom layer is built the same way, so it can be mixed
freely with the built-in layers, and trained like any other model.

This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_mnist, batch_iterator

# %%
# MNIST dataset parameters.
num_classes = 10  # 0 to 9 digits
num_features = 784  # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 500
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
## Create a custom layer

Build a custom layer with inner-variables.
"""

# Create a custom layer, extending Flax 'Module' class.
# Layer compute: y = relu(W * x + b)
class CustomLayer1(nnx.Module):

    # Layer arguments. Flax layers create their weights right away, so the
    # number of input features must be given (there is no lazy 'build').
    def __init__(self, num_inputs, num_units, rngs):
        # Create weight variables for this layer. Wrapping an array in
        # 'nnx.Param' marks it as a trainable parameter of the layer.
        # 'rngs.params()' provides a new random key for the initialization.
        self.weight = nnx.Param(
            0.05 * jax.random.normal(rngs.params(), (num_inputs, num_units)))
        self.bias = nnx.Param(jnp.zeros(num_units))

    # Set forward pass (a 'Param' can be used like a regular array).
    def __call__(self, inputs):
        x = jnp.matmul(inputs, self.weight)
        x = x + self.bias
        return nnx.relu(x)

"""
## Create another custom layer

Build another custom layer with inner Flax 'layers'.
"""

# Create a custom layer, extending Flax 'Module' class.
# Custom layer: 2 Fully-Connected layers with residual connection.
class CustomLayer2(nnx.Module):

    # Layer arguments.
    def __init__(self, num_units, rngs):
        # Note: a custom layer can include any other Flax layers, their
        # weights are then part of the custom layer.
        self.inner_layer1 = nnx.Linear(num_units, 1, rngs=rngs)
        self.inner_layer2 = nnx.Linear(1, num_units, rngs=rngs)

    def __call__(self, inputs):
        x = self.inner_layer1(inputs)
        x = nnx.relu(x)
        x = self.inner_layer2(x)
        return x + inputs

"""
## Build Model
"""

# Create Flax Model.
class CustomNet(nnx.Module):

    def __init__(self, rngs):
        # Use custom layers created above.
        self.layer1 = CustomLayer1(num_features, 64, rngs=rngs)
        self.layer2 = CustomLayer2(64, rngs=rngs)
        # And a built-in layer, for the output.
        self.out = nnx.Linear(64, num_classes, rngs=rngs)

    # Set forward pass.
    def __call__(self, x):
        x = self.layer1(x)
        x = nnx.relu(x)
        x = self.layer2(x)
        # Return the raw scores (logits): the loss function applies softmax.
        return self.out(x)

# Build neural network model.
custom_net = CustomNet(rngs=nnx.Rngs(0))

# %%
# Display the shape of every model parameter: the weights of the custom
# layers are tracked like those of the built-in layers.
for path, param in nnx.to_flat_state(nnx.state(custom_net, nnx.Param)):
    print('/'.join(map(str, path)), param.shape)

# %%
# Cross-Entropy loss function.
# Note that this will apply 'softmax' to the logits.
def cross_entropy(y_pred, y_true):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y_true)
    return jnp.mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = jnp.equal(jnp.argmax(y_pred, 1), y_true)
    return jnp.mean(correct_prediction.astype(jnp.float32))

# Adam optimizer.
optimizer = nnx.Optimizer(custom_net, optax.adam(learning_rate), wrt=nnx.Param)

# %%
# Optimization process.
# @nnx.jit compiles the function with XLA, so it runs much faster
# (see '1_Introduction/transformations.py').
@nnx.jit
def run_optimization(custom_net, optimizer, x, y):
    # Define the loss as a function of the model, to differentiate it.
    def loss_fn(custom_net):
        pred = custom_net(x)
        return cross_entropy(pred, y)

    # Compute gradients (for the custom layers weights too).
    gradients = nnx.grad(loss_fn)(custom_net)

    # Update the model parameters following gradients.
    optimizer.update(custom_net, gradients)

# %%
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Get the next batch of data.
    batch_x, batch_y = next(train_data)
    # Run the optimization to update the model parameters.
    run_optimization(custom_net, optimizer, batch_x, batch_y)

    if step % display_step == 0:
        pred = custom_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
# Test model on validation set.
pred = custom_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
