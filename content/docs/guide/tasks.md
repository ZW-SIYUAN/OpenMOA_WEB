---
title: Tasks
weight: 2
---

OpenMOA provides reproducible baselines for the following open-environment streaming tasks where the feature space is non-stationary, growing, shrinking, or re-ordered over time.

### Task	

Streaming Anomaly Detection in Open Feature Spaces	
Detect anomalies while feature set expands or contracts; maintain low false-positive rate under concept drift.	
• Sparse Active Online Learning (IJCAI'25, default)


Fast Online CUR Decomposition under Varying Features	
Reconstruct and track the latent low-rank structure of a data stream when column (feature) set keeps changing.	
• ℓ1,∞-Mixed-Norm CUR (SDM'24)

Utilitarian Online Learning from Soft Sensing Streams	
Make real-time decisions when the incoming features are unreliable (soft) and the feature space is open.	
• Utilitarian OLLA (ICDM'24)

Dynamic Feature Selection & Model Adaptation	
Decide on-the-fly which new features to keep and how to update the model without full retraining.	
• Online Group LASSO wrapper



### Planned Extensions
Multilayer streaming graphs with dynamic node attributes.
Higher-order joint tensor factorization under open modes.
Federated OpenMOA: collaborative learning across institutions without sharing feature lists.
