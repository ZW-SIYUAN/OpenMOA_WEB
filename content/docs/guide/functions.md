---
title: Functions
weight: 1
---

## Functions
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


