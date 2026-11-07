""" Multi-Device Training Example.

Train a convolutional neural network on multiple devices (GPUs or TPUs)
with JAX, using data parallelism.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## Training with multiple devices

In this example, we are using data parallelism to split the training across multiple devices (e.g. GPUs). Each device has a full replica of the neural network model, and the weights are updated synchronously, once every device has processed its share of the data.

First, each device processes a distinct slice of the batch and computes the corresponding gradients, then, the gradients of all devices are averaged. The model weights are finally updated with the averaged gradients, identically on every device, to repeat the training process.

<img src="https://www.tensorflow.org/images/Parallelism.png" alt="Parallelism" style="width: 400px;"/>

## Data parallelism in JAX

In JAX, we don't need to write code for each device. We only describe how the arrays are laid out across the devices (their 'sharding'):
- the model weights are **replicated**: every device holds a full copy.
- every batch of data is **sharded**: it is split along the batch dimension, and every device holds one slice.

Then, the usual training step compiled with `jax.jit` automatically runs on all devices at once: each device computes the gradients on its own slice of data, and JAX averages the gradients across devices before updating the weights.

## CIFAR10 Dataset Overview

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

![CIFAR10 Dataset](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)

More info: https://www.cs.toronto.edu/~kriz/cifar.html
"""

# This example uses the GPUs of the machine. Without them, the line below
# splits the CPU into 8 simulated devices, so that it still runs (this must be
# done BEFORE importing jax).
import os
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_force_host_platform_device_count=8")

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_cifar10, batch_iterator

# %%
# CIFAR10 dataset parameters.
num_classes = 10  # total classes.
num_devices = len(jax.devices())

# Training parameters.
learning_rate = 0.001
# The batch is split equally between the devices (1024 examples per device).
# Note: Reduce batch size if you encounter OOM Errors.
if jax.default_backend() != "cpu":
    training_steps = 1000
    batch_size = 1024 * num_devices
else:
    # The simulated CPU devices are much slower than GPUs and TPUs, so they
    # get a smaller batch (32 examples per device) and a shorter training.
    training_steps = 300
    batch_size = 32 * num_devices
display_step = 20

# Network parameters.
conv1_filters = 64  # number of filters for 1st conv layer.
conv2_filters = 128  # number of filters for 2nd conv layer.
conv3_filters = 256  # number of filters for 3rd conv layer.
fc1_units = 1024  # number of neurons for 1st fully-connected layer.

# %%
# Prepare CIFAR10 data.
(x_train, y_train), (x_test, y_test) = load_cifar10()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.
y_train, y_test = np.reshape(y_train, (-1)), np.reshape(y_test, (-1))

# %%
# Shuffle and batch data (a simple data loader, see 'common/datasets.py').
train_data = batch_iterator(x_train, y_train, batch_size)

# %%
# Arrange all the devices in a 1-D 'mesh', with a single axis named 'batch'.
mesh = Mesh(jax.devices(), ('batch',))
# Replicated layout: every device holds a full copy of the array.
replicated = NamedSharding(mesh, PartitionSpec())
# Sharded layout: the first dimension of the array (the batch) is split
# across the devices of the 'batch' mesh axis.
data_sharding = NamedSharding(mesh, PartitionSpec('batch'))

# %%
# Example: split a batch of images across the devices.
batch_x = jax.device_put(x_train[:batch_size], data_sharding)
print("Batch of shape %s, split across %i devices:" % (batch_x.shape, num_devices))
for shard in batch_x.addressable_shards:
    # 'shard.index' is the part of the batch that this device holds.
    images = range(batch_size)[shard.index[0]]
    print("  %s holds images %i to %i, shape: %s"
          % (shard.device, images.start, images.stop - 1, shard.data.shape))

# %%
# Create Flax Model.
class ConvNet(nnx.Module):
    # Set layers.
    def __init__(self, rngs):
        # Convolution Layers with 64 filters and a kernel size of 3.
        self.conv1_1 = nnx.Conv(3, conv1_filters, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv1_2 = nnx.Conv(conv1_filters, conv1_filters, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)

        # Convolution Layers with 128 filters and a kernel size of 3.
        self.conv2_1 = nnx.Conv(conv1_filters, conv2_filters, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)
        self.conv2_2 = nnx.Conv(conv2_filters, conv2_filters, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)
        self.conv2_3 = nnx.Conv(conv2_filters, conv2_filters, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)

        # Convolution Layers with 256 filters and a kernel size of 3.
        self.conv3_1 = nnx.Conv(conv2_filters, conv3_filters, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)
        self.conv3_2 = nnx.Conv(conv3_filters, conv3_filters, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)
        self.conv3_3 = nnx.Conv(conv3_filters, conv3_filters, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)

        # Fully connected layer (input: the 8x8x256 feature maps, flattened).
        self.fc1 = nnx.Linear(8 * 8 * conv3_filters, fc1_units, rngs=rngs)
        # Apply Dropout (only in training mode, see 'model.train()').
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

        # Output layer, class prediction.
        self.out = nnx.Linear(fc1_units, num_classes, rngs=rngs)

    # Set forward pass.
    def __call__(self, x):
        x = nnx.relu(self.conv1_1(x))
        x = nnx.relu(self.conv1_2(x))
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2_1(x))
        x = nnx.relu(self.conv2_2(x))
        x = nnx.relu(self.conv2_3(x))
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv3_1(x))
        x = nnx.relu(self.conv3_2(x))
        x = nnx.relu(self.conv3_3(x))
        # Flatten the data to a 1-D vector for the fully connected layer.
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc1(x))
        x = self.dropout(x)
        # Return the raw scores (logits): the loss function applies softmax.
        return self.out(x)

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

# %%
# Build convnet.
conv_net = ConvNet(rngs=nnx.Rngs(0))
# Adam optimizer.
optimizer = nnx.Optimizer(conv_net, optax.adam(learning_rate), wrt=nnx.Param)

# %%
# Copy the model and optimizer states (weights, Adam statistics, ...) on
# every device.
state = nnx.state((conv_net, optimizer))
state = jax.device_put(state, replicated)
nnx.update((conv_net, optimizer), state)

# %%
# Optimization process.
# Note that this is the same code as for a single device! As the model is
# replicated and the batch is sharded, '@nnx.jit' compiles it to run on all
# devices: each device computes the gradients on its slice of the batch,
# the gradients are averaged across devices, and each device applies the
# same update to its copy of the model.
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
conv_net.train()  # Training mode: enable dropout.
ts, last_step = time.time(), 0
for step in range(1, training_steps + 1):
    # Get the next batch of data, and split it across the devices.
    batch_x, batch_y = next(train_data)
    batch_x = jax.device_put(batch_x, data_sharding)
    batch_y = jax.device_put(batch_y, data_sharding)
    # Run the optimization to update the model parameters.
    run_optimization(conv_net, optimizer, batch_x, batch_y)

    if step % display_step == 0 or step == 1:
        # JAX runs computations asynchronously: wait for the training steps to
        # finish before measuring the training speed.
        jax.block_until_ready(nnx.state(conv_net))
        speed = batch_size * (step - last_step) / (time.time() - ts)
        conv_net.eval()  # Evaluation mode: disable dropout.
        pred = conv_net(batch_x)
        conv_net.train()
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f, speed: %f examples/sec" % (step, loss, acc, speed))
        ts, last_step = time.time(), step

# %%
# Test model on validation set, a few images at a time (to limit memory
# usage), split across the devices. A batch must split evenly across the
# devices, so its size is a multiple of the number of devices.
conv_net.eval()
eval_batch_size = 128 * num_devices
correct_predictions = 0
for i in range(0, len(x_test), eval_batch_size):
    batch_x = x_test[i:i + eval_batch_size]
    num_images = len(batch_x)
    # Pad the last (smaller) batch with blank images, so it still splits
    # evenly, and ignore their predictions.
    padding = eval_batch_size - num_images
    batch_x = np.concatenate([batch_x, np.zeros((padding,) + batch_x.shape[1:], np.float32)])
    pred = conv_net(jax.device_put(batch_x, data_sharding))[:num_images]
    correct_predictions += np.sum(np.argmax(pred, axis=1) == y_test[i:i + eval_batch_size])
print("Test Accuracy: %f" % (correct_predictions / len(x_test)))

"""
Note: on GPUs, the training goes through the training set about 80 times (1000 steps of 4096 images). The accuracy on the training batches gets close to 1, but the test accuracy is much lower: the model learned the training images by heart, instead of learning to recognize new images (it 'overfits'). Image augmentation (see the 'Image Transformation' example) would help.

## Single device vs multiple devices

To see the speedup given by multiple devices, let's time a few training steps of a new model on a single device, then on all the devices, with the same batch.
"""

# %%
# Time a few training steps of a new model on the given devices, and return
# the training speed (examples/sec).
def training_speed(devices, num_steps=20):
    # The same layouts as above, on the given devices only.
    mesh = Mesh(devices, ('batch',))
    replicated = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec('batch'))
    # A new model and optimizer, copied on every device.
    model = ConvNet(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    nnx.update((model, optimizer), jax.device_put(nnx.state((model, optimizer)), replicated))
    # The same batch for every step, split across the devices.
    batch_x = jax.device_put(x_train[:batch_size], data_sharding)
    batch_y = jax.device_put(y_train[:batch_size], data_sharding)
    # Warm-up: the first step compiles the training step for these devices.
    run_optimization(model, optimizer, batch_x, batch_y)
    jax.block_until_ready(nnx.state(model))
    ts = time.time()
    for step in range(num_steps):
        run_optimization(model, optimizer, batch_x, batch_y)
    # Wait for the training steps to finish before stopping the clock.
    jax.block_until_ready(nnx.state(model))
    return batch_size * num_steps / (time.time() - ts)

# %%
print("Single device training speed: %f examples/sec" % training_speed(jax.devices()[:1]))
print("Multi device training speed (%i devices): %f examples/sec"
      % (num_devices, training_speed(jax.devices())))
if jax.default_backend() == "cpu":
    print("Note: the simulated CPU devices share the same CPU cores, so they give "
          "no speedup. Run this example with multiple GPUs (or TPUs) to see it.")
