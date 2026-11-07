""" Notebook generator for the JAX examples.

Generates the executed .ipynb notebooks under notebooks/ from
the example scripts, so the two formats never drift apart: edit the .py
files, then re-run this tool (do not hand-edit the notebooks).

For each script: the module docstring becomes a markdown title cell, other
top-level string literals become markdown cells, the code is split into
cells at '# %%' marker lines (the "percent" cell format, also understood by
editors like VS Code to run a script cell by cell), and __file__-based paths
are rewritten for the notebook working directory (notebooks/<category>/
sits one level deeper than <category>/). Each notebook is then executed with
a real kernel so its outputs are saved.

Usage: run with a Python that has the examples requirements and the
notebook tooling (nbformat, nbclient, ipykernel) installed:

    pip install -r requirements.txt nbformat nbclient ipykernel
    python common/build_notebooks.py [filters]

Optional [filters] are substrings selecting which scripts to rebuild.
A full build re-trains every model, but skips the 6_Hardware notebooks.

The 6_Hardware notebooks show the speedup of multiple devices, which the
simulated CPU devices don't give: build them on a machine with GPUs (2 GPUs
for the basics, 4 for the training, like the TensorFlow examples), by naming
them, e.g.:

    CUDA_VISIBLE_DEVICES=0,1 python common/build_notebooks.py multi_device_basics
    CUDA_VISIBLE_DEVICES=0,1,2,3 python common/build_notebooks.py multi_device_training

Author: Aymeric Damien
Project: https://github.com/aymericdamien/JAX-Examples/
"""

import ast
import os
import re
import sys
import time

import nbformat as nbf
from nbclient import NotebookClient

SRC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DST = os.path.join(SRC, "notebooks")

SCRIPTS = [
    "1_Introduction/basic_operations.py",
    "1_Introduction/transformations.py",
    "2_BasicModels/linear_regression.py",
    "2_BasicModels/logistic_regression.py",
    "2_BasicModels/nearest_neighbor.py",
    "2_BasicModels/kmeans.py",
    "2_BasicModels/random_forest.py",
    "2_BasicModels/gradient_boosted_trees.py",
    "2_BasicModels/word2vec.py",
    "3_NeuralNetworks/neural_network.py",
    "3_NeuralNetworks/neural_network_raw.py",
    "3_NeuralNetworks/convolutional_network.py",
    "3_NeuralNetworks/convolutional_network_raw.py",
    "3_NeuralNetworks/recurrent_network.py",
    "3_NeuralNetworks/bidirectional_rnn.py",
    "3_NeuralNetworks/dynamic_rnn.py",
    "3_NeuralNetworks/autoencoder.py",
    "3_NeuralNetworks/variational_autoencoder.py",
    "3_NeuralNetworks/gan.py",
    "3_NeuralNetworks/dcgan.py",
    "4_Utils/save_restore_model.py",
    "4_Utils/build_custom_layers.py",
    "4_Utils/tensorboard.py",
    "5_DataManagement/load_data.py",
    "5_DataManagement/build_an_image_dataset.py",
    "5_DataManagement/array_records.py",
    "5_DataManagement/image_transformation.py",
    "6_Hardware/multi_device_basics.py",
    "6_Hardware/multi_device_training.py",
]

PREREQ_MD = ["0_Prerequisite/ml_introduction.md",
             "0_Prerequisite/mnist_dataset_intro.md"]

# notebooks/<cat>/ sits one level deeper than <cat>/, so every path that was
# anchored to the script location gains one extra "..", and __file__ (absent
# in notebooks) becomes os.getcwd() (nbclient runs each notebook from its
# own directory, like jupyter does).
FILE_ANCHOR = re.compile(
    r'os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\), "\.\."')
REPLACEMENT = 'os.path.join(os.getcwd(), "..", ".."'

# A line holding only this marker starts a new code cell.
CELL_MARKER = "# %%"


def rewrite_paths(source):
    source = FILE_ANCHOR.sub(REPLACEMENT, source)
    # Fallback for joins split across lines: the script directory's parent
    # (the repository root) is "../.." from the notebook cwd, so dirname(__file__)
    # itself maps to "..": join associativity keeps downstream ".." hops
    # correct, e.g. join(join(cwd, ".."), "..", "out") == cwd/../../out.
    source = source.replace(
        "os.path.dirname(os.path.abspath(__file__))",
        'os.path.join(os.getcwd(), "..")')
    assert "__file__" not in source, "unhandled __file__ reference"
    return source


def docstring_to_markdown(text):
    lines = text.strip().splitlines()
    title = lines[0].strip().rstrip(".")
    body = "\n".join(lines[1:]).strip()
    # Turn the "Author:/Project:" trailer into a markdown list.
    body = re.sub(r"Author: (.*)", r"- Author: \1", body)
    body = re.sub(r"Project: (\S+)", r"- Project: [\1](\1)", body)
    return "# %s\n\n%s" % (title, body)


def py_to_notebook(py_path):
    source = rewrite_paths(open(py_path).read())
    tree = ast.parse(source)
    lines = source.splitlines()

    # Top-level string expressions become markdown cells.
    md_ranges = {}
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            md_ranges[node.lineno - 1] = (node.end_lineno - 1, node.value.value)

    cells = []
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            cells.append(nbf.v4.new_code_cell(text))
        buf.clear()

    i = 0
    first_string = True
    while i < len(lines):
        if i in md_ranges:
            end, text = md_ranges[i]
            flush()
            if first_string and i == 0:
                cells.append(
                    nbf.v4.new_markdown_cell(docstring_to_markdown(text)))
            else:
                cells.append(nbf.v4.new_markdown_cell(text.strip()))
            first_string = False
            i = end + 1
            continue
        line = lines[i]
        if line.strip() == CELL_MARKER:
            flush()  # The marker starts a new cell (and is not kept).
        else:
            buf.append(line)
        i += 1
    flush()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3",
                              "language": "python"}
    nb.metadata.language_info = {"name": "python"}
    return nb


def md_to_notebook(md_path):
    text = open(md_path).read()
    # One directory deeper: fix relative links out of the category folder.
    text = text.replace("](../common/", "](../../common/")
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(text.strip())])
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3",
                              "language": "python"}
    return nb


def main():
    only = sys.argv[1:]
    for rel in PREREQ_MD:
        out = os.path.join(DST, rel.replace(".md", ".ipynb"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        nbf.write(md_to_notebook(os.path.join(SRC, rel)), out)
        print("%-55s markdown-only" % rel, flush=True)

    failures = []
    for rel in SCRIPTS:
        if only and not any(o in rel for o in only):
            continue
        # A full build skips the 6_Hardware notebooks, so that it doesn't
        # replace their GPU outputs with outputs from a machine without GPUs.
        if not only and rel.startswith("6_Hardware/"):
            print("%-55s skipped (build it on GPUs, by name)" % rel, flush=True)
            continue
        out = os.path.join(DST, rel.replace(".py", ".ipynb"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        nb = py_to_notebook(os.path.join(SRC, rel))
        t0 = time.time()
        try:
            client = NotebookClient(
                nb, timeout=900, kernel_name="python3",
                resources={"metadata": {"path": os.path.dirname(out)}})
            client.execute()
            status = "OK"
            nbf.write(nb, out)
        except Exception as e:
            # Keep the previous notebook: a failed one must not be committed.
            status = "FAIL: %s" % type(e).__name__
            failures.append((rel, str(e)[:400]))
        print("%-55s %-8s %6.1fs  (%d cells)"
              % (rel, status, time.time() - t0, len(nb.cells)), flush=True)

    print("\n==== SUMMARY ====")
    print("failures: %d" % len(failures))
    for rel, err in failures:
        print("--", rel, "--\n", err)
    # A non-zero exit code tells scripts and CI that a notebook failed.
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
