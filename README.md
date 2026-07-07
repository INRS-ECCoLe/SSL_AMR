# Physics-Informed Representations for Semi-Supervised Automatic Modulation Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper: **"Physics-Informed Representations for Semi-Supervised Automatic Modulation Classification"** (IEEE).

## 📖 Overview

Deep learning is the dominant approach for Automatic Modulation Classification (AMC), but acquiring labeled data in non-cooperative RF environments is highly expensive. This project introduces a **Physics-Informed (PI) input representation**—augmenting raw I/Q samples with instantaneous amplitude ($A$) and phase ($\phi$)—to enhance Semi-Supervised Learning (SSL) under severe label scarcity (5% - 20%).

By utilizing SNR-localized self-training with Softmax pseudo-labeling, the PI representation enables recurrent models to achieve near fully-supervised performance while completely preventing the training collapse typically observed in traditional LSTMs under high label scarcity.

### Evaluated Architectures (SSL)
This repository isolates the differences between Traditional (I/Q only) and Physics-Informed (I/Q + A + $\phi$) representations across three architectures:
* **CNN**: `ssl_CNN.py` vs. `ssl_PICNN.py`
* **BiLSTM**: `SSL_LSTM.py` vs. `SSL_PILSTM.py`
* **PETCGDNN**: `ssl_PETCGDNN.py` vs. `ssl_PIPETCGDNN.py`

---

## 🚀 Setup and Installation (`requirements.txt`)

To ensure reproducibility, please install the following required Python packages. You can save the block below as `requirements.txt` and install it, or install the packages directly.

```text
torch>=2.0.0
numpy>=1.21.0
h5py>=3.1.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
pandas>=1.3.0
