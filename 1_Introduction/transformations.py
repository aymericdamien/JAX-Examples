""" JAX Transformations Example.

Get started with the core JAX function transformations:

- `jax.grad`: automatic differentiation (compute gradients).
- `jax.jit`: just-in-time compilation with XLA (make code faster).
- `jax.vmap`: automatic vectorization (apply a function over a batch).

A transformation takes a Python function and returns a new function. They
can be freely combined, e.g. `jax.jit(jax.vmap(jax.grad(f)))`.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import time

import jax
import jax.numpy as jnp

# %%
# jax.grad: automatic differentiation.

# A simple function: f(x) = x^3 + 2x.
def f(x):
    return x ** 3 + 2. * x

# %%
# jax.grad(f) is a new function that computes the derivative of f:
# f'(x) = 3x^2 + 2.
df = jax.grad(f)
print("f(2.) =", f(2.))
print("f'(2.) =", df(2.))

# %%
# Transformations can be combined: the derivative of the derivative is the
# second derivative, f''(x) = 6x.
print("f''(2.) =", jax.grad(jax.grad(f))(2.))

# %%
# For a function with several inputs, 'argnums' selects which inputs to
# differentiate with respect to.
def g(x, y):
    return 3. * x ** 2 + 2. * x * y

dg_dx, dg_dy = jax.grad(g, argnums=(0, 1))(2., 3.)
print("dg/dx =", dg_dx, ", dg/dy =", dg_dy)

# %%
# Inputs can also be a dictionary (or list, tuple...) of arrays, like the
# weights of a model: the gradient then has the same structure.
def model_loss(params, x):
    return jnp.sum((params['w'] * x + params['b']) ** 2)

params = {'w': 2., 'b': 1.}
grads = jax.grad(model_loss)(params, 3.)
print("dloss/dw =", grads['w'], ", dloss/db =", grads['b'])

# %%
# jax.jit: compilation with XLA.

# A function doing many small operations.
def slow_function(x):
    for _ in range(50):
        x = 0.5 * x + 0.1 * jnp.sin(x)
    return x

# %%
# jax.jit(slow_function) compiles the whole function into a single fast
# program. The compilation happens on the first call, so call it once first.
fast_function = jax.jit(slow_function)
x = jnp.ones([100, 100])
y = fast_function(x)

# %%
# Compare the speed. JAX runs computations asynchronously:
# 'block_until_ready()' waits for the result, so we measure the real time.
start = time.time()
for _ in range(100):
    slow_function(x).block_until_ready()
print("without jit: %f sec" % (time.time() - start))

start = time.time()
for _ in range(100):
    fast_function(x).block_until_ready()
print("with jit: %f sec" % (time.time() - start))

# %%
# jit can also be applied as a decorator.
@jax.jit
def selu(x, alpha=1.67, scale=1.05):
    return scale * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

print(selu(jnp.arange(-2., 3.)))

# %%
# jax.vmap: automatic vectorization.

# A function written for a single example (a vector of 3 values).
w = jnp.array([2., -1., 0.5])

def predict(x):
    return jnp.dot(w, x)

print("one example:", predict(jnp.array([1., 2., 3.])))

# %%
# jax.vmap(predict) is a new function that works on a batch of examples
# (it maps 'predict' over the first dimension of its input).
batch = jnp.array([[1., 2., 3.],
                   [4., 5., 6.],
                   [7., 8., 9.]])
print("batch of examples:", jax.vmap(predict)(batch))

# %%
# Combining transformations.

# The derivative of f, computed for a batch of inputs, and compiled.
batched_df = jax.jit(jax.vmap(jax.grad(f)))
print("f'(x) for x in [0, 1, 2, 3]:", batched_df(jnp.array([0., 1., 2., 3.])))
