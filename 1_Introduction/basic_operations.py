""" Basic Array Operations Example.

Basic array operations using JAX.

`jax.numpy` provides the same functions as NumPy, but runs them on CPU, GPU
or TPU. This example also covers two things that differ from NumPy: arrays
are immutable, and random numbers need an explicit random key.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import jax
import jax.numpy as jnp
import numpy as np

# %%
# Define array constants.
a = jnp.array(2)
b = jnp.array(3)
c = jnp.array(5)

# Access arrays value.
print("a =", a)
print("b =", b)
print("c =", c)

# %%
# Various array operations.
# Note: Arrays also support python operators (+, *, ...).
add = jnp.add(a, b)
sub = jnp.subtract(a, b)
mul = jnp.multiply(a, b)
div = jnp.divide(a, b)

# Access arrays value.
print("add =", add)
print("sub =", sub)
print("mul =", mul)
print("div =", div)

# %%
# Some more operations.
mean = jnp.mean(jnp.array([a, b, c]))
sum = jnp.sum(jnp.array([a, b, c]))

# Access arrays value.
print("mean =", mean)
print("sum =", sum)

# %%
# Matrix multiplications.
matrix1 = jnp.array([[1., 2.], [3., 4.]])
matrix2 = jnp.array([[5., 6.], [7., 8.]])

product = jnp.matmul(matrix1, matrix2)

# %%
# Display Array.
print(product)

# %%
# Convert Array to Numpy.
print(np.asarray(product))

# %%
# JAX functions also accept NumPy arrays as inputs (but not Python lists:
# convert them with 'jnp.array' first).
print(jnp.matmul(matrix1, np.array([[5., 6.], [7., 8.]])))

# %%
# Arrays are immutable: they can't be modified in place.
x = jnp.array([1, 2, 3, 4, 5])
try:
    x[0] = 10
except TypeError:
    print("x[0] = 10 is not allowed: JAX arrays are immutable.")

# %%
# Instead, 'x.at[index].set(value)' returns a new array with the update.
y = x.at[0].set(10)
print("x =", x)
print("y =", y)

# %%
# Random numbers.
# Every random function takes a random 'key' as input. The same key always
# gives the same numbers.
key = jax.random.key(0)
print(jax.random.normal(key, [2, 2]))
print(jax.random.normal(key, [2, 2]))

# %%
# To get new random numbers, split the key into new, independent keys.
key1, key2 = jax.random.split(key)
print(jax.random.normal(key1, [2, 2]))
print(jax.random.uniform(key2, [2, 2]))
