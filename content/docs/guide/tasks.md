---
title: Tasks
weight: 2
---

List of Benchmark Tasks

## Task-1 Efficient & Interpretable Online Learning in Open, High-Dimensional Feature Spaces (SOOFS & RSOL)
### (1) Real-World Scenario
Imagine a city-scale disaster-monitoring system that continuously ingests data from crowd-sensing smartphones, IoT sensors, and social-media feeds.
- Devices come and go: a new iPhone introduces a fresh sensor modality; an old Android goes offline, removing its features; a viral hashtag spikes and vanishes within minutes.
- The feature space is open (arbitrarily evolving) and often high-dimensional (millions of hashtags, sensor channels, or user contexts).
- The system must predict in real time (e.g., “Is this region in danger?”) while keeping the model small enough to fit in edge-device memory and transparent enough for emergency managers to audit.

### (2) Formal Task Definition
Input
- Streaming instances: ${(x_t, y_t) \mid t \in [T]}$
  - $x_t \in ℝ^{d_t}$ where $d_t$ can grow or shrink at each round.
  - $y_t \in {+1, –1}$ (binary label, e.g., “danger” vs “safe”).

Knowledge Constraints
- Only one pass over the data (online).
- Model size must stay sub-linear in the total number of ever-seen features.

Objective

At each round t, output a sparse weight vector $w_t \in ℝ^{d_t}$ minimizing

$$Regret_T = Σ_{t=1}^T ℓ_t(w_t)  +  λ‖W_t‖_{1,∞} or ‖W_t‖_{1,2} $$

where
- $ℓ_t$ is the hinge loss: $\max(0,\; 1 - y_t(\mathbf{w}_t^\top \mathbf{x}_t))$.
- $‖W_t‖_{1,∞} or ‖W_t‖_{1,2}$ enforces row-wise sparsity (entire features pruned).
- Expected Influence: maximize prediction accuracy while keeping non-zero rows ≤ budget k (user-defined memory limit).

### (3) Loss Function & Optimization Target
Minimize sub-linear regret over T rounds:

$$min_{w_t}  Σ_{t=1}^T ℓ_t(w_t)   s.t.   ‖W‖_{1,∞} ≤ k$$

- Closed-form PA update handles dynamic dimension alignment.
- Online CUR decomposes the sliding-window weight matrix to actively retain the most informative instances, ensuring both efficiency and interpretability (original features retained or dropped en bloc).

───────────────────────────────

## Task-2 Efficient & Interpretable Online Learning in Open, Unknown Categories appear at any time Feature Spaces
## (1) Problem Statement
In feature-evolving and class-open online environments (e.g., semiconductor wafer inspection lines), sensors may be added or removed over time, and previously unseen defect types can emerge at any moment. OWSS seeks an online learner that, at each round t, receives a sample

$$(x_t, y_t), x_t \in ℝ^{d_t}, y_t ∈ Y_k ∪ Y_u$$,

where dₜ can increase (new sensors) or decrease (obsolete sensors), and Yᵤ denotes unknown classes. The learner must decide on the fly whether to output a class label or to abstain.

## (2) Objective
Minimize the cumulative loss over a horizon T:

$$\min_{h_1,\dots,h_T}\;
\sum_{t=1}^{T}
\Bigl[\ell\bigl(y_t,h_t(x_t)\bigr)\cdot\mathbb{I}\bigl(h_t\ \text{predicts}\bigr)+c(x_t)\cdot\mathbb{I}\bigl(h_t\ \text{abstains}\bigr)\Bigr]$$,

subject to
- Universal representation: zₜ = φ(xₜ) ∈ ℝ^{k} with fixed k, updated via bipartite graph–GCN.
- Abstention rule: r(zₜ) ≥ 0 ⇒ predict; r(zₜ) < 0 ⇒ abstain.
- Online update: parameters (Θ, {fᵢ}, r) are updated once per sample without revisiting prior data.

## (3) Baselines
• closed-world online: OCO, OSLMF
• offline outlier detection: ECOD, LUNAR
• graph offline: GraSSNet
• open-world with pre-training: ORCA

───────────────────────────────

### Task-3 Utilitarian Online Regression from Open-World Soft Sensing (UORS)

Given:
- Multi-source soft sensor stream X_t = [x_t^(1), …, x_t^(M)], where any source may go offline or be newly added at any time;
- The target variable y_t arrives with a delay only after the user submits a "utility query";
- The global utility function U(ŷ, y) is specified online by the user (e.g., energy consumption-accuracy tradeoff).

Goal:
Maintain a regressor h_t to ensure cumulative utility
Maximize ∑_{τ=1}^t U(h_τ(x_τ), y_τ), and ensure the model is robust to heterogeneous sensor drifts.

Constraints:
- The sensor dimension can be extended to 10^4, but only m_t ≪ M active sources are observed in each round;
- Support utility function hot swapping without retraining.

Baseline implementation:
- UOL-SS (ICDM’24) – Online Mirror Descent + Dynamic Weight Sharing + Sparse Group Lasso;
- ORFF-Util – a weighted version of Random Feature Forests, supporting online utility re-weighting;
- Ridge-Restart – Classical Ridge Regression + Periodic Restart Baseline.

Evaluation indicators:
- Cumulative Utility Regret vs. offline oracle
- Sensor Robustness = AUROC (Fault Detection)
- Model Update Latency

───────────────────────────────

### Task-4 Online Deep Representation Learning with Evolving Feature Spaces (ODR-EFS)

Given:
- High-dimensional stream data X_t ∈ ℝ^(n_t×d_t), where d_t varies over time;
- An optional semi-supervised signal (with weak labels for some samples).

Goal:
Joint learning
(1) Dynamic feature encoder ϕ_t: ℝ^{d_t} → ℝ^k, where k is fixed;
(2) The online classification/regression head ψ_t ensures that the accuracy of downstream tasks remains unchanged while keeping the model capacity under control.

Constraints:
- The number of encoder parameters grows sublinearly with d_t;
- Support catastrophic forgetting inhibition.

Baseline implementation:
- DV-RNN (TKDE’23) – Deep Variational Recurrent Neural Network + Scalable Feature Gating;
- ORFF-Encoder – Single-layer ORFF as a lightweight encoder;
- MLP-Reinit – Reinitialize the fully connected layer every time the dimension is expanded.

Evaluation indicators:
- Downstream task Accuracy@t
- Forgetting Measure = Accuracy@t – max_{τ≤t} Accuracy@τ
- Feature Expansion Cost (FLOPs)

───────────────────────────────

### Task-5 Online Random Feature Forest for Varying Feature Streams (ORFF-VF)

Given:
- Streaming binary classification data (x_t, y_t) containing only partial feature observations;
- Tree nodes and random features can be dynamically added or deleted in each round.

Goal:
Maintain the error of the ORFF classifier f_t to be ≤ ε, and ensure the model size is ≤ B MB.
Constraints:
- Tree depth ≤ 8, minimum sample size for leaf nodes = 10;
- Single-sample prediction latency ≤ 1 ms.

Baseline implementation:
- ORFF-VS (AAAI’22) – Original algorithm + Adaptive feature pool;
- ORFF-Budget – a pruned version with budget constraints;
- Hoeffding-Tree – a classic stream decision tree baseline.

Evaluation indicators:
- Error Rate@t
- Model Size (MB)
- Feature Utilization Ratio = Number of active features / Total number of observed features

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
