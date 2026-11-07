""" Multi-Device Basics Example.

A simple example to introduce computation on multiple devices with JAX:
compute A^n + B^n on a single device, then split the work across 2 devices
(device 0 computes A^n while device 1 computes B^n), and compare timings.

In JAX, every processing unit (CPU, GPU or TPU) is a 'device'. An array
lives on a device, and computation follows the data: an operation runs on
the device where its inputs are stored.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
This tutorial requires your machine to have 2 devices (e.g. 2 GPUs). So that
it runs on any machine, we also ask JAX to split the CPU into 8 simulated
devices, used when there are fewer than 2 GPUs or TPUs.
"""

# This example uses 2 GPUs. With fewer of them, the line below splits the CPU
# into 8 simulated devices, so that it still runs (this must be done BEFORE
# importing jax).
import os
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_force_host_platform_device_count=8")

import time

import jax
import jax.numpy as jnp
import numpy as np

# %%
# List the processing units available to JAX: the GPUs or TPUs if there are
# any, the CPU otherwise. This example needs 2 devices: with a single GPU or
# TPU, we use the simulated CPU devices instead.
devices = jax.devices()
if len(devices) < 2:
    devices = jax.devices("cpu")
for device in devices:
    print("Device %i: %s" % (device.id, device))

# %%
# Power of the matrices to compute (A^n and B^n).
n = 10
# Size of the (square) matrices. The CPU is much slower than GPUs and TPUs,
# so it gets smaller matrices.
matrix_size = 2000 if devices[0].platform == "cpu" else 10000

# %%
# Place an array on a given device, with 'jax.device_put'.
x = jnp.arange(4.)
x0 = jax.device_put(x, devices[0])
x1 = jax.device_put(x, devices[1])
print("x0 is stored on:", x0.devices())
print("x1 is stored on:", x1.devices())

# %%
# Computation follows the data: 'x1 * 2' runs on device 1, where x1 lives.
y1 = x1 * 2
print("x1 * 2 is computed (and stored) on:", y1.devices())

"""
## Example: compute A^n + B^n on 2 devices
"""

# Create random large matrices.
A = np.random.rand(matrix_size, matrix_size).astype('float32')
B = np.random.rand(matrix_size, matrix_size).astype('float32')

# Matrix power: M^n = M x M^(n-1), computed with n-1 matrix multiplications.
def matpow(M, n):
    if n == 1:
        return M
    else:
        return jnp.matmul(M, matpow(M, n - 1))

# Compute A^n and B^n, each on the device where its input lives, then sum
# them on device 0 (arrays must be on the same device to be added).
def matpow_sum(a, b):
    a_n = matpow(a, n)
    b_n = matpow(b, n)
    return jax.device_put(a_n, devices[0]) + jax.device_put(b_n, devices[0])

"""
## Single device computing
"""

# Place both matrices on device 0: it computes A^n, then B^n.
a = jax.device_put(A, devices[0])
b = jax.device_put(B, devices[0])

# Warm-up run: the first run of an operation on a device includes some
# one-time setup (compilation), which we don't want to measure.
matpow_sum(a, b).block_until_ready()

t1_1 = time.time()
result = matpow_sum(a, b)
# JAX runs computations asynchronously: wait for the result to be ready
# before stopping the clock.
result.block_until_ready()
t2_1 = time.time()

"""
## Multi device computing
"""

# Place A on device 0 and B on device 1: the two matrix powers are computed
# at the same time, each by its own device.
a = jax.device_put(A, devices[0])
b = jax.device_put(B, devices[1])

# Warm-up run.
matpow_sum(a, b).block_until_ready()

t1_2 = time.time()
result = matpow_sum(a, b)
result.block_until_ready()
t2_2 = time.time()

# %%
print("Single device computation time: %f sec" % (t2_1 - t1_1))
print("Multi device computation time: %f sec" % (t2_2 - t1_2))
if devices[0].platform == "cpu":
    print("Note: the simulated CPU devices share the same CPU cores, so they give "
          "no speedup. Run this example with 2 GPUs (or TPUs) to see it.")

"""
With 2 GPUs, both matrix powers are computed at the same time, so the multi
device version is about twice as fast.
"""
