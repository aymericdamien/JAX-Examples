# JAX Examples

This tutorial was designed for easily diving into [JAX](https://docs.jax.dev/), through examples. For readability, it includes both notebooks and source codes with explanation.

It is suitable for beginners who want to find clear and concise examples about JAX. Besides the 'raw' JAX implementations (`jax.numpy`, `jax.grad`, `jax.jit`, `jax.vmap`, ...), you can also find the most common libraries of the JAX ecosystem:
- [Flax](https://flax.readthedocs.io/) (NNX API): neural network layers and models (`nnx.Linear`, `nnx.Conv`, `nnx.LSTMCell`, ...).
- [Optax](https://optax.readthedocs.io/): optimizers (SGD, Adam, ...) and loss functions.
- [Orbax](https://orbax.readthedocs.io/): save and restore models.
- [Grain](https://google-grain.readthedocs.io/): input data pipelines.

## Tutorial index

#### 0 - Prerequisite
- [Introduction to Machine Learning](0_Prerequisite/ml_introduction.md) ([notebook](notebooks/0_Prerequisite/ml_introduction.ipynb)).
- [Introduction to MNIST Dataset](0_Prerequisite/mnist_dataset_intro.md) ([notebook](notebooks/0_Prerequisite/mnist_dataset_intro.ipynb)).

#### 1 - Introduction
- **Basic Operations** ([notebook](notebooks/1_Introduction/basic_operations.ipynb)) ([code](1_Introduction/basic_operations.py)). A simple example that cover JAX basic operations.
- **JAX Transformations** ([notebook](notebooks/1_Introduction/transformations.ipynb)) ([code](1_Introduction/transformations.py)). Get started with JAX core transformations: `grad` (automatic differentiation), `jit` (compilation) and `vmap` (vectorization).

#### 2 - Basic Models
- **Linear Regression** ([notebook](notebooks/2_BasicModels/linear_regression.ipynb)) ([code](2_BasicModels/linear_regression.py)). Implement a Linear Regression with JAX.
- **Logistic Regression** ([notebook](notebooks/2_BasicModels/logistic_regression.ipynb)) ([code](2_BasicModels/logistic_regression.py)). Implement a Logistic Regression with JAX.
- **Nearest Neighbor** ([notebook](notebooks/2_BasicModels/nearest_neighbor.ipynb)) ([code](2_BasicModels/nearest_neighbor.py)). Implement Nearest Neighbor algorithm with JAX.
- **K-Means** ([notebook](notebooks/2_BasicModels/kmeans.ipynb)) ([code](2_BasicModels/kmeans.py)). Build a K-Means classifier with JAX.
- **Random Forest** ([notebook](notebooks/2_BasicModels/random_forest.ipynb)) ([code](2_BasicModels/random_forest.py)). Build a Random Forest classifier with JAX.
- **GBDT (Gradient Boosted Decision Trees)** ([notebook](notebooks/2_BasicModels/gradient_boosted_trees.ipynb)) ([code](2_BasicModels/gradient_boosted_trees.py)). Implement a Gradient Boosted Decision Trees with JAX to predict house value using Boston Housing dataset.
- **Word2Vec (Word Embedding)** ([notebook](notebooks/2_BasicModels/word2vec.ipynb)) ([code](2_BasicModels/word2vec.py)). Build a Word Embedding Model (Word2Vec) from Wikipedia data, with JAX.

#### 3 - Neural Networks
##### Supervised

- **Simple Neural Network** ([notebook](notebooks/3_NeuralNetworks/neural_network.ipynb)) ([code](3_NeuralNetworks/neural_network.py)). Use Flax 'layers' and 'model' API to build a simple neural network to classify MNIST digits dataset.
- **Simple Neural Network (low-level)** ([notebook](notebooks/3_NeuralNetworks/neural_network_raw.ipynb)) ([code](3_NeuralNetworks/neural_network_raw.py)). Raw implementation of a simple neural network to classify MNIST digits dataset.
- **Convolutional Neural Network** ([notebook](notebooks/3_NeuralNetworks/convolutional_network.ipynb)) ([code](3_NeuralNetworks/convolutional_network.py)). Use Flax 'layers' and 'model' API to build a convolutional neural network to classify MNIST digits dataset.
- **Convolutional Neural Network (low-level)** ([notebook](notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)) ([code](3_NeuralNetworks/convolutional_network_raw.py)). Raw implementation of a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** ([notebook](notebooks/3_NeuralNetworks/recurrent_network.ipynb)) ([code](3_NeuralNetworks/recurrent_network.py)). Build a recurrent neural network (LSTM) to classify MNIST digits dataset, using Flax 'layers' and 'model' API.
- **Bi-directional Recurrent Neural Network (LSTM)** ([notebook](notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)) ([code](3_NeuralNetworks/bidirectional_rnn.py)). Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset, using Flax 'layers' and 'model' API.
- **Dynamic Recurrent Neural Network (LSTM)** ([notebook](notebooks/3_NeuralNetworks/dynamic_rnn.ipynb)) ([code](3_NeuralNetworks/dynamic_rnn.py)). Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of variable length, using Flax 'layers' and 'model' API.

##### Unsupervised
- **Auto-Encoder** ([notebook](notebooks/3_NeuralNetworks/autoencoder.ipynb)) ([code](3_NeuralNetworks/autoencoder.py)). Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **Variational Auto-Encoder** ([notebook](notebooks/3_NeuralNetworks/variational_autoencoder.ipynb)) ([code](3_NeuralNetworks/variational_autoencoder.py)). Build a variational auto-encoder (VAE), to encode and generate images from noise.
- **GAN (Generative Adversarial Networks)** ([notebook](notebooks/3_NeuralNetworks/gan.ipynb)) ([code](3_NeuralNetworks/gan.py)). Build a Generative Adversarial Network (GAN) to generate images from noise.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](notebooks/3_NeuralNetworks/dcgan.ipynb)) ([code](3_NeuralNetworks/dcgan.py)). Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

#### 4 - Utilities
- **Save and Restore a model** ([notebook](notebooks/4_Utils/save_restore_model.ipynb)) ([code](4_Utils/save_restore_model.py)). Save and Restore a model with JAX, using Orbax.
- **Build Custom Layers & Modules** ([notebook](notebooks/4_Utils/build_custom_layers.ipynb)) ([code](4_Utils/build_custom_layers.py)). Learn how to build your own layers / modules and integrate them into Flax Models.
- **Tensorboard** ([notebook](notebooks/4_Utils/tensorboard.ipynb)) ([code](4_Utils/tensorboard.py)). Track and visualize neural network computation graph, metrics, weights and more using TensorBoard.

#### 5 - Data Management
- **Load and Parse data** ([notebook](notebooks/5_DataManagement/load_data.ipynb)) ([code](5_DataManagement/load_data.py)). Build efficient data pipeline with Grain (Numpy arrays, Images, CSV files, custom data, ...).
- **Build an image dataset** ([notebook](notebooks/5_DataManagement/build_an_image_dataset.ipynb)) ([code](5_DataManagement/build_an_image_dataset.py)). Build your own images dataset from image folders, and train a model on it.
- **Build and Load ArrayRecords** ([notebook](notebooks/5_DataManagement/array_records.ipynb)) ([code](5_DataManagement/array_records.py)). Convert data into ArrayRecord format (the record format of Grain), and load them.
- **Image Transformation (i.e. Image Augmentation)** ([notebook](notebooks/5_DataManagement/image_transformation.ipynb)) ([code](5_DataManagement/image_transformation.py)). Apply various image augmentation techniques with JAX, to generate distorted images for training.

#### 6 - Hardware
- **Multi-Device Basics** ([notebook](notebooks/6_Hardware/multi_device_basics.ipynb)) ([code](6_Hardware/multi_device_basics.py)). A simple example to introduce computation on multiple devices (GPUs, TPUs) with JAX.
- **Multi-Device Training** ([notebook](notebooks/6_Hardware/multi_device_training.ipynb)) ([code](6_Hardware/multi_device_training.py)). Train a convolutional neural network with multiple devices (GPUs, TPUs) on CIFAR-10 dataset.

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples (with the small helper [`common/datasets.py`](common/datasets.py), files are cached in a `data/` folder).
MNIST is a database of handwritten digits, for a quick description of that dataset, you can check [this notebook](notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

Official Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).

## Installation

To download all the examples, simply clone this repository:
```
git clone https://github.com/aymericdamien/JAX-Examples
```

To run them, you also need JAX and a few libraries of its ecosystem. To install them:
```
pip install -r requirements.txt
```

or (with NVIDIA GPU support):
```
pip install -r requirements.txt "jax[cuda13]"
```

For older NVIDIA GPUs or drivers (e.g. V100 GPUs, or drivers older than 580), use `"jax[cuda12]"` instead.

For more details about JAX installation (GPU, TPU, ...), you can check [JAX Installation Guide](https://docs.jax.dev/en/latest/installation.html).

## Others

This is the JAX version of [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples). Every example follows the same structure, so you can also compare both side by side.

