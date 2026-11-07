""" TensorBoard Example.

Graph, Loss, Accuracy & Weights visualization using TensorBoard and JAX.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/).

TensorBoard displays the content of log files ('event files'). Here they
are written with tensorboardX, a small library to write TensorBoard logs
from any Python program (TensorFlow is not required).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import os
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorboardX import SummaryWriter

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_mnist, batch_iterator

# %%
# Path to save logs into.
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
logs_path = os.path.join(out_dir, "tensorboard_logs", "example")

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
# new key for every weight.
key_h1, key_h2, key_logits = jax.random.split(jax.random.key(0), 3)

# A random value generator to initialize weights.
def random_normal(key, shape):
    return 0.05 * jax.random.normal(key, shape)

# All the model parameters, gathered in a dictionary.
params = {
    'weights': {
        'h1_weights': random_normal(key_h1, [num_features, n_hidden_1]),
        'h2_weights': random_normal(key_h2, [n_hidden_1, n_hidden_2]),
        'logits_weights': random_normal(key_logits, [n_hidden_2, num_classes]),
    },
    'biases': {
        'h1_bias': jnp.zeros([n_hidden_1]),
        'h2_bias': jnp.zeros([n_hidden_2]),
        'logits_bias': jnp.zeros([num_classes]),
    },
}

# %%
# Construct model.
def neural_net(params, x):
    weights, biases = params['weights'], params['biases']
    # Hidden fully connected layer with 128 neurons.
    layer_1 = jnp.add(jnp.matmul(x, weights['h1_weights']), biases['h1_bias'])
    # Apply sigmoid to layer_1 output for non-linearity.
    layer_1 = jax.nn.sigmoid(layer_1)
    # Hidden fully connected layer with 256 neurons.
    layer_2 = jnp.add(jnp.matmul(layer_1, weights['h2_weights']), biases['h2_bias'])
    # Apply sigmoid to layer_2 output for non-linearity.
    layer_2 = jax.nn.sigmoid(layer_2)
    # Output fully connected layer with a neuron for each class.
    out_layer = jnp.matmul(layer_2, weights['logits_weights']) + biases['logits_bias']
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

    # Compute gradients.
    gradients = jax.grad(loss_fn)(params)

    # Update weights/biases following gradients.
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# %%
# Visualize weights & biases as histogram in Tensorboard.
def summarize_weights(step):
    # Names like 'h1/weights' group the histograms by layer in TensorBoard.
    for w in params['weights']:
        summary_writer.add_histogram(w.replace('_', '/'), np.asarray(params['weights'][w]), step)
    for b in params['biases']:
        summary_writer.add_histogram(b.replace('_', '/'), np.asarray(params['biases'][b]), step)

# %%
# Remove the logs of a previous run (if any), to start from a clean directory.
shutil.rmtree(logs_path, ignore_errors=True)
# Create a Summary Writer to log the metrics to Tensorboard.
summary_writer = SummaryWriter(logs_path)

# %%
# Log the computation graph.
# JAX traces the model function into a 'jaxpr': the list of primitive
# operations that XLA compiles. It is logged as text (TensorBoard 'Text'
# tab), indented by 4 spaces to be displayed as code.
graph = jax.make_jaxpr(neural_net)(params, x_train[:batch_size])
# Convert it to plain text (without the colors used to print it in a terminal).
graph_text = graph.pretty_print(use_color=False)
print(graph_text)
summary_writer.add_text('graph', '\n'.join('    ' + line for line in graph_text.splitlines()), 0)

# %%
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Get the next batch of data.
    batch_x, batch_y = next(train_data)
    # Run the optimization to update weights/biases.
    params, opt_state = run_optimization(params, opt_state, batch_x, batch_y)

    if step % display_step == 0:
        pred = neural_net(params, batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

        # Write loss/acc metrics & weights to Tensorboard every few steps,
        # to avoid storing too much data.
        summary_writer.add_scalar('loss', float(loss), step)
        summary_writer.add_scalar('accuracy', float(acc), step)
        summarize_weights(step)

# %%
# Write all pending logs to disk.
summary_writer.close()

# %%
# Test model on validation set.
pred = neural_net(params, x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

"""
### Run Tensorboard

To run tensorboard, install it (it does not require TensorFlow) and run the following command in your terminal, from the root of the repository:
```
pip install tensorboard
tensorboard --logdir=out/tensorboard_logs
```

And then connect your web browser to: [http://localhost:6006](http://localhost:6006)

The 'Scalars' tab shows the loss and accuracy curves, the 'Histograms' and 'Distributions' tabs show how the weights evolve during training, and the 'Text' tab shows the model graph (jaxpr).
"""

"""
![tensorboard1](../../resources/img/tensorboard1.png)
"""

"""
![tensorboard2](../../resources/img/tensorboard2.png)
"""

"""
![tensorboard3](../../resources/img/tensorboard3.png)
"""
