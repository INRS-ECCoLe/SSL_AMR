## Methodology and Performance Evaluation for Semi‑Supervised Learning Based Modulation Signal Classification

### 1. Introduction

This section presents a refined methodological framework for evaluating semi‑supervised learning (SSL) in automatic modulation recognition (AMR) using the RadioML 2018.01A (RML2018) dataset. The objective is to address labeled‑data scarcity in realistic wireless environments by leveraging a controlled self‑training strategy based on high‑confidence pseudo‑labels. Multiple neural architectures are evaluated under identical conditions to ensure a fair and reproducible comparison.

### 2. Dataset and Preprocessing

The RML2018 dataset contains complex baseband I/Q samples spanning 24 modulation classes across a wide SNR range. To emulate practical deployment conditions, the dataset is randomly shuffled and partitioned as follows:

* **Global Split:** 80% of samples are allocated for training and 20% for final evaluation.
* **Labeled vs. Unlabeled Split:** Within the training portion, only 30% of samples retain labels for supervised learning, while the remaining 70% form an unlabeled pool for SSL.
* **SNR‑Focused Filtering:** The unlabeled pool is further constrained to low‑SNR samples (−2, 0, and 2 dB). This design choice forces the SSL mechanism to operate near decision boundaries, where classification uncertainty is highest, rather than exploiting trivially separable high‑SNR data.

### 3. Model Architectures

Five deep learning architectures are considered, each selected to represent a distinct modeling paradigm:

* **CNN:** A four‑layer 2D convolutional network serving as a baseline spatial feature extractor.
* **LSTM:** Recurrent models designed to capture temporal dependencies in sequential I/Q samples.
* **MCLDNN:** A hybrid architecture combining parallel 1D and 2D convolutions with an LSTM backend to jointly model spatial and temporal features.
* **PET‑CGDNN:** A phase‑aware network that explicitly estimates and compensates phase offsets prior to CNN‑GRU processing.

Model‑specific input formatting is applied to ensure architectural consistency without altering the underlying signal information.

### 4. Semi‑Supervised Learning Strategy

The SSL framework follows an iterative self‑training procedure applied independently to each model:

1. **Base Training:** Each model is first trained on the limited labeled subset using supervised learning. Early stopping and adaptive learning‑rate scheduling are employed to prevent overfitting.
2. **Confidence‑Based Pseudo‑Labeling:** The trained model infers labels for unlabeled samples. Predictions exceeding predefined confidence thresholds (0.60–0.90) are selected.
3. **Quality Monitoring:** Although not used during training, an oracle comparison with hidden ground truth is performed to estimate pseudo‑label reliability and detect potential confirmation bias.
4. **Retraining:** The model is retrained on a combined dataset of original labeled samples and newly pseudo‑labeled data. This process is repeated for five iterations per confidence threshold.

### 5. Experimental Configuration

All experiments use a consistent setup: Adam optimizer with an initial learning rate of 0.001, batch size of 400, 150 epochs for base training, and 50 epochs for SSL retraining. Training and inference are accelerated using CUDA‑enabled GPUs.

### 6. Evaluation Metrics and Outputs

Performance is assessed using overall accuracy, SNR‑specific accuracy, confusion matrices, and average inference time. Results are serialized into structured JSON files, enabling detailed post‑hoc analysis of performance trends across confidence thresholds and SSL iterations.

### 7. Summary

The revised pipeline emphasizes clarity, reproducibility, and robustness analysis. By concentrating SSL on low‑SNR regimes, the framework directly evaluates whether pseudo‑labeling can improve modulation classification in challenging noise conditions, providing meaningful insights into the practical viability of semi‑supervised AMR systems.
