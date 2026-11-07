""" Word2Vec (Word Embedding) Example.

Implement Word2Vec algorithm to compute vector representations of words,
with JAX. This example is using a small chunk of Wikipedia articles to
train from.

More info: [Mikolov, Tomas et al. "Efficient Estimation of Word Representations in Vector Space.", 2013](https://arxiv.org/pdf/1301.3781.pdf)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import collections
import os
import random
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import load_text8

# %%
# Training Parameters.
# The training is short (30000 steps): a large batch and a large learning
# rate make the most of it.
learning_rate = 1.0
batch_size = 1024
num_steps = 30000  # Train longer (e.g. 300000 steps) to get better embeddings.
display_step = 3000
eval_step = 10000

# Evaluation Parameters.
eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']

# Word2Vec Parameters.
embedding_size = 200  # Dimension of the embedding vector.
max_vocabulary_size = 50000  # Total number of different words in the vocabulary.
min_occurrence = 10  # Remove all words that does not appears at least n times.
skip_window = 3  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# %%
# Download a small chunk of Wikipedia articles collection (~17M words).
# Text has already been processed (lowercase words separated by spaces).
text_words = load_text8()

# %%
# Build the dictionary and replace rare words with UNK token.
count = [('UNK', -1)]
# Retrieve the most common words.
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
# Remove samples with less than 'min_occurrence' occurrences.
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        # The collection is ordered, so stop when 'min_occurrence' is reached.
        break
# Compute the vocabulary size.
vocabulary_size = len(count)
# Assign an id to each word.
word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])

# %%
data_index = 0
# Generate training batch for the skip-gram model.
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size), dtype=np.int32)
    # get window size (words left and right + current one).
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch.
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# %%
# Random values are generated from a random 'key', which is split to get a
# new key for every use.
key = jax.random.key(0)
key_embedding, key_nce, key = jax.random.split(key, 3)

# %%
# A random value generator to initialize weights (small values train faster here).
def random_normal(key, shape):
    return 0.1 * jax.random.normal(key, shape)

# All the model parameters, gathered in a dictionary.
params = {
    # The embedding matrix: each row represents a word embedding vector.
    'embedding': random_normal(key_embedding, [vocabulary_size, embedding_size]),
    # The weights and biases of the NCE loss (one row/value per word).
    'nce_weights': random_normal(key_nce, [vocabulary_size, embedding_size]),
    'nce_biases': jnp.zeros([vocabulary_size]),
}

def get_embedding(params, x):
    # Lookup the corresponding embedding vectors for each sample in X.
    return params['embedding'][x]

def nce_loss(params, x_embed, y, key):
    """ A simple form of the Noise-Contrastive Estimation (NCE) loss, called
    'negative sampling'. Computing a softmax over the ~47,000 words of the
    vocabulary at every step would be slow. Instead, each input word only
    has to tell its true context word apart from a few random 'negative' words.
    (The full NCE loss also corrects every score for how often its word is
    sampled; negative sampling skips that correction, which doesn't matter
    to learn good word embeddings.)
    """
    # Sample 'num_sampled' negative words, shared by the whole batch. As the
    # vocabulary is sorted by frequency, a log-uniform distribution samples
    # frequent words (small ids) more often than rare ones.
    u = jax.random.uniform(key, [num_sampled])
    negative_words = jnp.floor(jnp.exp(u * jnp.log(vocabulary_size + 1.))).astype(jnp.int32) - 1
    # Score (logit) of each input word with its true context word.
    true_logits = jnp.sum(x_embed * params['nce_weights'][y], axis=1) + params['nce_biases'][y]
    # Scores of each input word with every negative word.
    negative_logits = (jnp.matmul(x_embed, params['nce_weights'][negative_words].T)
                       + params['nce_biases'][negative_words])
    # Binary classification: true context words should get label 1,
    # negative words label 0.
    true_loss = optax.sigmoid_binary_cross_entropy(true_logits, jnp.ones_like(true_logits))
    negative_loss = optax.sigmoid_binary_cross_entropy(negative_logits, jnp.zeros_like(negative_logits))
    # Compute the average NCE loss for the batch.
    return jnp.mean(true_loss + jnp.sum(negative_loss, axis=1))

# Evaluation.
def evaluate(params, x_embed):
    # Compute the cosine similarity between input data embedding and every embedding vectors.
    x_embed_norm = x_embed / jnp.linalg.norm(x_embed, axis=1, keepdims=True)
    embedding = params['embedding']
    embedding_norm = embedding / jnp.linalg.norm(embedding, axis=1, keepdims=True)
    return jnp.matmul(x_embed_norm, embedding_norm.T)

# Define the optimizer.
optimizer = optax.sgd(learning_rate)
# The optimizer state is stored outside of the optimizer.
opt_state = optimizer.init(params)

# %%
# Optimization process.
# @jax.jit compiles the function with XLA, so it runs much faster
# (see '1_Introduction/transformations.py'). The embedding matrices are big:
# 'donate_argnums' lets JAX reuse the memory of the old 'params' and
# 'opt_state' (arguments 0 and 1) to store the updated ones.
@jax.jit(donate_argnums=(0, 1))
def run_optimization(params, opt_state, x, y, key):
    # Define the loss as a function of the parameters, to differentiate it.
    def loss_fn(params):
        emb = get_embedding(params, x)
        return nce_loss(params, emb, y, key)

    # Compute the loss and its gradients.
    loss, gradients = jax.value_and_grad(loss_fn)(params)

    # Update the parameters following gradients.
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)

    # JAX arrays can't be modified in place: return the updated values.
    return params, opt_state, loss

# %%
# Words for testing.
x_test = np.array([word2id[w] for w in eval_words])

# Run training for the given number of steps.
for step in range(1, num_steps + 1):
    batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
    # Get a new random key for the negative sampling of this step.
    key, step_key = jax.random.split(key)
    params, opt_state, loss = run_optimization(params, opt_state, batch_x, batch_y, step_key)

    if step % display_step == 0 or step == 1:
        print("step: %i, loss: %f" % (step, loss))

    # Evaluation.
    if step % eval_step == 0 or step == 1:
        print("Evaluation...")
        sim = np.array(evaluate(params, get_embedding(params, x_test)))
        for i in range(len(eval_words)):
            top_k = 8  # number of nearest neighbors.
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % eval_words[i]
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, id2word[nearest[k]])
            print(log_str)
