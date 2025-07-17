---
title: Functions
weight: 1
---

### prepare
```yaml
from openmoa.data import load_real, load_synthetic
from openmoa.data.moa import generate_synthetic
from openmoa.preprocess import datastream_select, open_scaler, elastic_projection, 
from openmoa.model import (
    SOADLearner,                       # IJCAI‘25
    OCURSketch,                        # SDM‘24
    UORSRegressor,                     # ICDM‘24
    RSOLClassifier,                    # SDM‘23
    ODREncoder,                        # TKDE‘23
    ORFForest                          # AAAI‘22
)
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


