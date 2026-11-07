""" Load and Parse Data Example.

Build efficient input pipelines for loading data, with Grain. JAX itself
does not load data: Grain is the data loading library of the JAX ecosystem.
It reads records from a data source, and chains transformations (shuffle,
map, batch, ...) that are run in background threads, so that the next
batches are ready when the model needs them.

- Numpy Arrays
- CSV file
- Images
- Custom data from a Python class

For more information about creating and loading ArrayRecord files (the
record format used by Grain), see 'array_records.py'.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import csv
import os
import shutil
import string
import sys
import tarfile

import grain
import numpy as np
from PIL import Image

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import DATA_DIR, download

"""
### Load Numpy Arrays

Build a data pipeline over numpy arrays.
"""

# Create a toy dataset (even and odd numbers, with respective labels of 0 and 1).
evens = np.arange(0, 100, step=2, dtype=np.int32)
evens_label = np.zeros(50, dtype=np.int32)
odds = np.arange(1, 100, step=2, dtype=np.int32)
odds_label = np.ones(50, dtype=np.int32)
# Concatenate arrays
features = np.concatenate([evens, odds])
labels = np.concatenate([evens_label, odds_label])

# Load the data with Grain: any Python sequence (here a list of (feature, label)
# pairs) can be the source of a dataset.
data = grain.MapDataset.source(list(zip(features, labels)))
# Shuffle data (with a seed, for reproducibility).
data = data.shuffle(seed=0)
# Refill data indefinitely (re-shuffled at every epoch).
data = data.repeat()
# Batch data (aggregate records together).
data = data.batch(batch_size=4)

# %%
# A 'MapDataset' can be indexed and sliced like a list: take the first 5 batches.
for batch_x, batch_y in data[:5]:
    print(batch_x, batch_y)

# %%
# Note: If you are planning on calling multiple time,
# you can use the iterator way. 'to_iter_dataset' also prefetches batches
# (pre-load batches in background threads for faster consumption).
ite_data = iter(data.to_iter_dataset(grain.ReadOptions(num_threads=2, prefetch_buffer_size=2)))
for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)

for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)

"""
### Load CSV files

Build a data pipeline from features stored in a CSV file. For this example, Titanic dataset will be used as a toy dataset stored in CSV format.

#### Titanic Dataset

survived|pclass|name|sex|age|sibsp|parch|ticket|fare
--------|------|----|---|---|-----|-----|------|----
1|1|"Allen, Miss. Elisabeth Walton"|female|29|0|0|24160|211.3375
1|1|"Allison, Master. Hudson Trevor"|male|0.9167|1|2|113781|151.5500
0|1|"Allison, Miss. Helen Loraine"|female|2|1|2|113781|151.5500
0|1|"Allison, Mr. Hudson Joshua Creighton"|male|30|1|2|113781|151.5500
...|...|...|...|...|...|...|...|...
"""

# Download Titanic dataset (in csv format).
titanic_path = download(
    ["https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv"],
    "titanic_dataset.csv")

# %%
# Load Titanic dataset.
# Original features: survived,pclass,name,sex,age,sibsp,parch,ticket,fare
# Select specific columns: survived,pclass,name,sex,age,fare
column_to_use = [0, 1, 2, 3, 4, 8]
record_types = [int, int, str, str, float, float]

# Load the whole dataset file, and parse each line with Python's csv module.
records = []
with open(titanic_path) as f:
    reader = csv.reader(f, skipinitialspace=True)
    # Skip header.
    next(reader)
    for row in reader:
        # Keep the selected columns, converted to their type.
        record = tuple(t(row[c]) for c, t in zip(column_to_use, record_types))
        records.append(record)

data = grain.MapDataset.source(records)
# Shuffle data.
data = data.shuffle(seed=0)
# Refill data indefinitely.
data = data.repeat()
# Batch data (aggregate records together).
data = data.batch(batch_size=2)

# %%
for survived, pclass, name, sex, age, fare in data[:1]:
    print(survived)
    print(pclass)
    print(name)
    print(sex)
    print(age)
    print(fare)

"""
### Load Images

Build a data pipeline by loading images from disk. For this example, Oxford Flowers dataset will be used.
"""

# Download Oxford 17 flowers dataset
archive_path = download(["https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"],
                        "17flowers.tgz")
# Extract archive (images are stored in a 'jpg' folder).
flowers_dir = os.path.join(DATA_DIR, "17flowers")
if not os.path.exists(flowers_dir):
    # Extract to a temporary folder, renamed once complete: an interrupted
    # extraction is then not mistaken for a complete one.
    shutil.rmtree(flowers_dir + ".tmp", ignore_errors=True)
    with tarfile.open(archive_path) as t:
        t.extractall(flowers_dir + ".tmp", filter="data")
    os.rename(flowers_dir + ".tmp", flowers_dir)

# %%
# Write a dataset file, listing every image with its label: the 17 flower
# species are stored in order, 80 images each.
with open(os.path.join(flowers_dir, "jpg", "dataset.csv"), 'w') as f:
    c = 0
    for i in range(1360):
        image_path = os.path.join(flowers_dir, "jpg", "image_%04i.jpg" % (i + 1))
        f.write("%s,%i\n" % (image_path, c))
        if (i + 1) % 80 == 0:
            c += 1

# %%
# Load Images
with open(os.path.join(flowers_dir, "jpg", "dataset.csv")) as f:
    dataset_file = f.read().splitlines()

# Load the whole dataset file, and slice each line.
data = grain.MapDataset.source(dataset_file)
# Shuffle data.
data = data.shuffle(seed=0)
# Refill data indefinitely.
data = data.repeat()

# Load and pre-process images.
def load_image(path):
    # Read image from path, and decode the jpeg image (with Pillow).
    image = Image.open(path).convert('RGB')
    # Resize images to a common size of 256x256.
    image = image.resize((256, 256))
    # Convert to an array of float32 values in [0, 255].
    image = np.array(image, np.float32)
    # Rescale values to [-1, 1].
    image = image / 127.5 - 1.
    return image

# Decode each line from the dataset file.
def parse_records(line):
    # File is in csv format: "image_path,label_id".
    image_path, image_label = line.split(',')
    # Apply the function to load images.
    image = load_image(image_path)
    return image, int(image_label)

# Use 'map' to apply the above functions to every record.
data = data.map(parse_records)

# Batch data (aggregate images-array together).
data = data.batch(batch_size=2)
# Prefetch batch: images are loaded and decoded in 4 parallel threads.
data = data.to_iter_dataset(grain.ReadOptions(num_threads=4, prefetch_buffer_size=4))

# %%
batch_x, batch_y = next(iter(data))
print("images:", batch_x.shape, batch_x.dtype, "min: %f, max: %f" % (batch_x.min(), batch_x.max()))
print("labels:", batch_y)

"""
### Load custom data from a Python class

Grain does not read Python generators: to shuffle the data (or to resume an interrupted
training), it needs random access to the records. So custom data comes from a 'data source':
any Python object with a `__len__` (number of records) and a `__getitem__` (return the record
at a given index) method, which can load or generate each record on the fly.
"""

# Create a dummy data source.
class RandomFeatures:
    # Total number of records.
    def __len__(self):
        return 1000

    # Generate the record number 'index'.
    def __getitem__(self, index):
        # A random number generator seeded with the index: the same index
        # always returns the same record.
        rng = np.random.default_rng(index)
        # Return a random string, a random vector, and a random int.
        random_string = ''.join(rng.choice(list(string.ascii_letters), size=4))
        return random_string, rng.uniform(size=4).astype(np.float32), int(rng.integers(0, 10))

# %%
# Load the data source with Grain.
data = grain.MapDataset.source(RandomFeatures())
# Shuffle data.
data = data.shuffle(seed=0)
# Refill data indefinitely.
data = data.repeat()
# Batch data (aggregate records together).
data = data.batch(batch_size=4)

# %%
# Display data.
for batch_str, batch_vector, batch_int in data[:5]:
    print(batch_str, batch_vector, batch_int)
