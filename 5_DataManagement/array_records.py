""" Create and Load ArrayRecords Example.

A simple example to parse a dataset into ArrayRecord format, and then read
that dataset with Grain.

ArrayRecord is the file format used by Grain (the data loading library of
the JAX ecosystem) to store large datasets: a file containing a sequence of
records (each one a sequence of bytes), where any record can be read
directly from its index. That 'random access' lets Grain shuffle the whole
dataset, and read it from multiple threads or machines in parallel.

In this example, the Titanic Dataset (in CSV format) will be used as a toy
dataset, for parsing all the dataset features into ArrayRecord format, and
then building an input pipeline that can be used for training models.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

"""
## Titanic Dataset

The titanic dataset is a popular dataset for ML that provides a list of all passengers onboard the Titanic, along with various features such as their age, sex, class (1st, 2nd, 3rd)... And if the passenger survived the disaster or not.

It can be used to see that even though some luck was involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class...

#### Overview
survived|pclass|name|sex|age|sibsp|parch|ticket|fare
--------|------|----|---|---|-----|-----|------|----
1|1|"Allen, Miss. Elisabeth Walton"|female|29|0|0|24160|211.3375
1|1|"Allison, Master. Hudson Trevor"|male|0.9167|1|2|113781|151.5500
0|1|"Allison, Miss. Helen Loraine"|female|2|1|2|113781|151.5500
0|1|"Allison, Mr. Hudson Joshua Creighton"|male|30|1|2|113781|151.5500
...|...|...|...|...|...|...|...|...


#### Variable Descriptions
```
survived        Survived
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
```
"""

import csv
import json
import os
import sys

import grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

# Make the shared helpers of the 'common' folder importable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.datasets import download

# %%
# Download Titanic dataset (in csv format).
titanic_path = download(
    ["https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv"],
    "titanic_dataset.csv")

# %%
# The ArrayRecord file will be written in the 'out' folder.
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
os.makedirs(out_dir, exist_ok=True)
record_path = os.path.join(out_dir, "titanic_dataset.array_record")

"""
### Create ArrayRecords
"""

# Generate a record, parsing all features of the dataset.
# A record is just bytes: here, the features are stored in a dictionary,
# serialized to text with JSON, then encoded to bytes.
def convert_to_record(survived, pclass, name, sex, age, sibsp, parch, ticket, fare):
    features = {
        'survived': survived,
        'pclass': pclass,
        'name': name,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'ticket': ticket,
        'fare': fare,
    }
    return json.dumps(features).encode('utf-8')

# %%
# Open dataset file.
with open(titanic_path) as f:
    # Output ArrayRecord file. 'group_size:1' stores each record separately,
    # so that any record can be read directly (fast random access).
    w = ArrayRecordWriter(record_path, 'group_size:1')
    # Generate a record for all row in our dataset.
    # CSV reader will read and parse all rows.
    reader = csv.reader(f, skipinitialspace=True)
    for i, record in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        survived, pclass, name, sex, age, sibsp, parch, ticket, fare = record
        # Parse each csv row to a record using the above function.
        example = convert_to_record(int(survived), int(pclass), name, sex, float(age),
                                    int(sibsp), int(parch), ticket, float(fare))
        # Write the serialized record to the ArrayRecord file.
        w.write(example)
    w.close()

"""
### Load ArrayRecords
"""

# Build features template, with types.
features = {
    'survived': np.int64,
    'pclass': np.int64,
    'name': str,
    'sex': str,
    'age': np.float32,
    'sibsp': np.int64,
    'parch': np.int64,
    'ticket': str,
    'fare': np.float32,
}

# %%
# Load ArrayRecord data.
filenames = [record_path]
data = grain.MapDataset.source(grain.sources.ArrayRecordDataSource(filenames))
print("Number of records:", len(data))

# Parse features, using the above template.
def parse_record(record):
    example = json.loads(record)
    return {name: dtype(example[name]) for name, dtype in features.items()}
# Apply the parsing to each record from the dataset.
data = data.map(parse_record)

# Shuffle data.
data = data.shuffle(seed=0)
# Refill data indefinitely.
data = data.repeat()
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batches in background threads for faster consumption).
data = data.to_iter_dataset(grain.ReadOptions(num_threads=2, prefetch_buffer_size=2))

# %%
# Dequeue data and display.
record = next(iter(data))
print(record['survived'])
print(record['name'])
print(record['fare'])
