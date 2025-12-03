---
title: Tutorials
weight: 1
---

### prepare
```yaml
from openmoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import (
    FESLClassifier,                       # NIPS‘2017
    FOBOSClassifier,                        # JMLR‘2009
    FTRLClassifier,                     # AISTATS‘2011
    OASFClassifier,                    # BigData‘2024
    RSOLClassifier,                        # SDM‘2024
    ORF3VClassifier,                       # AAAI'2022
    OVFMClassifier,                        # ICDM'2021
    OSLMFClassifier,                      # AAAI'2022
    OLD3SClassifier,                      # TKDE'2023
    OWSSClassifier                          # ICDM'2024
)
```

### Demo
```yaml
"""Demo script for FESL Classifier - Simplified Version"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FESLClassifier

# Load stream
elec_stream = Electricity()
schema = elec_stream.schema

print(f"Number of features: {schema.get_num_attributes()}")

# Create FESL learner
fesl_learner = FESLClassifier(
    schema=schema,
    s1_feature_indices=[0, 1, 2, 3],
    s2_feature_indices=[2, 3, 4, 5],
    overlap_size=100,
    switch_point=5000,
    ensemble_method="selection",
    learning_rate_scale=0.1,
    random_seed=42
)

# Run evaluation with progress bar
results = prequential_evaluation(
    stream=elec_stream,
    learner=fesl_learner,
    max_instances=10000,
    window_size=100,
    progress_bar=True  # Enable the progress bar
)

# Print final results
print(f"Ensemble Method: {fesl_learner.ensemble_method}")
print(f"\nFinal Accuracy: {results['cumulative'].accuracy():.2f}%")
print(f"Kappa: {results['cumulative'].kappa():.4f}")

```

### data loading functions
**1. openmoa.data.load_real()**

Function: Load real data stream flow.

Return: (X, y, feat_info)
- X: Scipy.sparse. csr_matrix has one sample per row, with dimensions increasing over time;
- y: Np.ndarray 0/1 label, where 1 indicates an exception;
- Feat_info: dict records the timestamp of adding/deleting features.
- Example:
```yaml
X, y, feat_info = load_real(real_dataset)
```

**2. openmoa.data.load_synthetic()**

Function: Load synthetic data stream flow.

Return: (X, y, feat_info)
- X: Scipy.sparse. csr_matrix has one sample per row, with dimensions increasing over time;
- y: Np.ndarray 0/1 label, where 1 indicates an exception;
- Feat_info: dict records the timestamp of adding/deleting features.
- Example:
```yaml
X, y, feat_info = load_synthetic(stnthetic_dataset)
```

### data preprocessing functions
**3. openmoa.preprocess datastream_select()**

Function: Select the corresponding data stream feature space to process the original dataset.

Return: (X, y, feat_info)
- X: Scipy.sparse. csr_matrix has one sample per row, with dimensions increasing over time;
- y: Np.ndarray 0/1 label, where 1 indicates an exception;
- Feat_info: dict records the timestamp of adding/deleting features.
- Example:
```yaml
X, y, feat_info = datastream_select(dataset)
```

**4. openmoa.preprocess.open_scaler()**
   
Function: Flow based robust normalizer (zero mean, unit variance), capable of incremental updates and automatically expanding mean/variance vectors as feature dimensions change.

Source: Universal module, but used as the default preprocessing in the robust experiment of SDM'23 RSOL.

API：
```yaml
scaler = open_scaler(with_centering=True, clip_outliers=3.0)
for batch in stream:
X_batch = scaler.partial_fit_transform(batch)
```

**5. openmoa.preprocess.elastic_projection()**

Function: Elastic sparse mapping, when new features appear, the online learning projection matrix compresses the original high-dimensional space to a fixed k-dimension while retaining anomaly discriminative power.

Source: Elastic Sparse Projection module of IJCAI'25 SOAD.

API：
```yaml
proj = elastic_projection(out_dim=128, l1_penalty=1e-3)
for X_batch in stream:
Z_batch = proj.partial_fit_transform(X_batch)   # Z ∈ R^{n×128}
```

### model functions
**6. openmoa.model.SOADLearner()**

Function: Sparse active online anomaly detector, implementing IJCAI'25 paper core algorithm:
- Integrate active selection (uncertainty+diversity+budget);
- Support incremental sparse weight updates in open feature spaces;
- Provide interfaces for. partial_fit (X, y=None) and. query (batch-budget).

Core parameters:
```yaml
soad = SOADLearner(
budget_per_round=50, #Up to 50 tags can be queried per round
l1_reg=1e-4, #Sparse regularization
Forget_rate=0.01 # Anti concept drift
)
```

**7. openmoa.model.OCURSketch()**

Function: Online CUR row and column sketch, targeting SDM'24 ℓ 1, ∞ - MXed Norm CUR algorithm:
- Row sparsity constraint and variable column dimension;
- press (rank=r, budget=b) returns (C, U, R) in one step.

Example:
```yaml
cur = OCURSketch(rank=50, row_sparsity=5)
for M_batch in matrix_stream:
C, U, R = cur.partial_compress(M_batch)
```

- '''Dataset Loading (openmoa.dataset.*)'''
  - ```stream_loader(), ds = om.dataset.stream_loader('synthetic_open', n_samples=1e6, feature_pace=500) ```	Return a streaming dataset with an infinite iterator based on its name, and specify a feature drift strategy
  - ```file_stream(path, fmt='csv|jsonl|parquet'), ds = om.dataset.file_stream('s3://bucket/log.parquet')```	Read real-time append files from local files or S3/HDFS	
  - ```kafka_stream(topic, brokers, schema), ds = om.dataset.kafka_stream('iot-sensor', brokers='kafka:9092')```	Directly consume from Kafka topics	
  - ```arxiv_open_citation_stream()```	Open feature drift flow reserved interface for real academic graphs

- '''Pre-processing (openmoa.preprocess.*)'''
  - ```adaptive_standardize(), ds = om.preprocess.adaptive_standardize(ds, alpha=0.01)```	Online mean variance standardization, supporting cold start for new features	
  - ```drift_detector(), flag = om.preprocess.drift_detector(window=1000)```Feature space drift detection (KL divergence/Maximum Mean Discrepancy)
  - ```feature_hashing(n_buckets), ds = om.preprocess.feature_hashing(ds, n_buckets=2**20)```	Hash dimension reduction when feature space explodes	
  - ```missing_value_imputer(strategy='mean|median|zero'), ds = om.preprocess.missing_value_imputer(strategy='zero')```	Fill in missing values online	


