---
title: Quick Start
date: 2025-07-11
weight: 1
---

## Installation

You may install OpenMOA library by pip
{{< highlight go "linenos=inline, hl_lines=3 6-8, style=emacs" >}}
pip install OpenMOA
{{< /highlight >}}


### The First Example

Run an end-to-end streaming-anomaly detection experiment on a synthetically drifting feature space.

Python
{{< highlight go "linenos=inline, hl_lines=3 6-8, style=emacs" >}}
import openmoa as om
from openmoa.dataset import stream_loader
from openmoa.preprocess import adaptive_standardize
from openmoa.model import SparseActiveOL  # IJCAI'25
from openmoa.eval import run
{{< /highlight >}}
# 1. create a streaming loader whose feature space grows/shrinks on-the-fly
{{< highlight go "linenos=inline, hl_lines=3 6-8, style=emacs" >}}
ds = stream_loader(name='synthetic_open',
                   n_samples=1_000_000,
                   feature_pace=500,   # new feature appears every 500 steps
                   anomaly_ratio=0.05)
{{< /highlight >}}
# 2. plug in the online learner
{{< highlight go "linenos=inline, hl_lines=3 6-8, style=emacs" >}}
learner = SparseActiveOL(
    alpha=0.01,           # sparsity regularizer (ℓ1,∞ mixed norm, SDM'24)
    budget=50,            # active query budget
    window=1000           # sliding window size
)
{{< /highlight >}}
# 3. run & collect results
{{< highlight go "linenos=inline, hl_lines=3 6-8, style=emacs" >}}
run(
    stream=ds,
    preprocess=adaptive_standardize,
    learner=learner,
    metrics=['AUC', 'F1', 'N_features'],
    output=['csv', 'fig', 'animation'],
    checkpoint_every=10000
)
{{< /highlight >}}

### Documentation Structure

[Configure your site name, description, and menu.](https://docs.hugoblox.com/tutorial/blog/)

### OL Models

[Edit the homepage and add your documentation pages.](https://docs.hugoblox.com/tutorial/blog/)

### Datasets

[Easily publish your site for free with GitHub Pages](https://docs.hugoblox.com/tutorial/blog/)
