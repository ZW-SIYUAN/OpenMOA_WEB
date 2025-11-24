---
title: Installation
date: 2025-07-11
weight: 1
---

This document describes how to install OpenMOA and its dependencies. OpenMOA is tested against Python 3.10, 3.11, and 3.12. Newer versions of Python will likely work but have yet to be tested.

1. Virtual Environment (Optional)
We recommend using a virtual environment to manage your Python environment. Miniconda is a good choice for managing Python environments. You can install Miniconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/main). Once you have Miniconda installed, you can create a new environment with:
```yaml
conda create -n openmoa python=3.11
conda activate openmoa
```
When your environment is activated, you can install OpenMOA by following the instructions below.

2. Java (Required)

CapyMOA requires Java to be installed and accessible in your environment. You can check if Java is installed by running the following command in your terminal:

```yaml
java -version
```

If Java is not installed, you can download OpenJDK (Open Java Development Kit) from [this link](https://openjdk.org/install/), or alternatively the Oracle JDK from [this link](https://www.oracle.com/java/). Linux users can also install OpenJDK using their distribution’s package manager.

Now that Java is installed, you should see an output similar to the following when you run the command :java -version

```yaml
openjdk version "17.0.9" 2023-10-17
OpenJDK Runtime Environment (build 17.0.9+8)
OpenJDK 64-Bit Server VM (build 17.0.9+8, mixed mode)
```

3. PyTorch (Required)

The CapyMOA algorithms using deep learning require PyTorch. It is not installed by default because different versions are required for different hardware. If you want to use these algorithms, follow the instructions [here](https://pytorch.org/get-started/locally/) to get the correct version for your hardware. Ensure that you install PyTorch in the same environment virtual environment where you want to install CapyMOA.

For CPU only, you can install PyTorch with:

```yaml
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. Install OpenMOA
5. 
```yaml
pip install openmoa
```

To verify your installation, run:

```yaml
python -c "import capymoa; print(capymoa.__version__)"
```

### The First Example

Run an end-to-end streaming-anomaly detection experiment on a synthetically drifting feature space.

Python
```yaml
import openmoa
from openmoa.datasets import Electricity             # import dataset
from openmoa.evaluation import prequential_evaluation # import evaluator
from openmoa.classifier import ORF3VClassifier       # import classifier
from openmoa.stream import EvolvingFeatureStream  # import stream wrapper
```

1. Create evolving feature stream with TDS pattern
```yaml
stream = EvolvingFeatureStream(
    base_stream=Electricity(),    # use Electricity dataset
    evolution_pattern="tds",      # time-dependent selection pattern
    d_max=6,                      # maximum feature space dimension
    total_instances=10000         # total instances to generate
)
```

2. Initialize ORF3V classifier
```yaml
learner = ORF3VClassifier(
    schema=stream.get_schema(),   # get data schema from stream
    n_stumps=20,                  # number of decision stumps
    alpha=0.3,                    # learning rate for weight updates
    grace_period=100              # collect stats before building forest
)
```

3. Evaluate with prequential method
```yaml
results = prequential_evaluation(
    stream=stream,                # data stream
    learner=learner,              # classifier to evaluate
    max_instances=10000,          # number of instances to process
    window_size=100               # sliding window size for metrics
)
```
