---
title: Functions
weight: 1
---

## Functions
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
1. openmoa.data.load_real()

Function: Load real data stream flow.

Return: (X, y, feat_info)
- X: Scipy.sparse. csr_matrix has one sample per row, with dimensions increasing over time;
- y: Np.ndarray 0/1 label, where 1 indicates an exception;
- Feat_info: dict records the timestamp of adding/deleting features.
- Example:
```yaml
X, y, feat_info = load_real(return_X_y=True)
```

2. openmoa.data.load_synthetic()

Function: Load synthetic data stream flow.

Return: (X, y, feat_info)
- X: Scipy.sparse. csr_matrix has one sample per row, with dimensions increasing over time;
- y: Np.ndarray 0/1 label, where 1 indicates an exception;
- Feat_info: dict records the timestamp of adding/deleting features.
- Example:
```yaml
X, y, feat_info = load_synthetic(return_X_y=True)
```

3. openmoa.preprocess datastream_select()

Function: Select the corresponding data stream feature space to process the original dataset.

Return: (X, y, feat_info)
- X: Scipy.sparse. csr_matrix has one sample per row, with dimensions increasing over time;
- y: Np.ndarray 0/1 label, where 1 indicates an exception;
- Feat_info: dict records the timestamp of adding/deleting features.
- Example:
```yaml
X, y, feat_info = load_synthetic(return_X_y=True)
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


