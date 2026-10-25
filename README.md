# SSL_AMR — Physics-Informed Representations for Semi-Supervised AMC

Code for the IEEE Signal Processing Letters submission
*"Physics-Informed Representations for Semi-Supervised Automatic Modulation
Classification"* (A. Mahabub and S. Vakili, INRS-ÉMT).

The repository trains nine models on RML2018.01A under full supervision and
under SNR-localized confidence-based self-training, and reproduces the paper's
figures and complexity table.

## Repository layout

| File | Purpose |
|---|---|
| `models.py` | All nine architectures and a `MODEL_REGISTRY` |
| `train.py` | Supervised baselines and SNR-localized self-training, all models |
| `per_model/train_<MODEL>.py` | The same pipeline, one self-contained file per model (no import of `models.py`) |
| `summarize_results.py` | Collects the JSON results into CSV tables |
| `benchmark_complexity.py` | Parameters, memory, GFLOPs and CPU latency (Table I) |
| `figures/plot_accuracy_vs_snr.py` | Fig. 1: accuracy vs. SNR, supervised and self-training |
| `figures/plot_confusion.py` | Fig. 2: PI-Bi-LSTM confusion matrices at 6 dB |
| `figures/plot_confusion_per_model.py` | Confusion matrices for any model and SNR, read from `train.py` results |
| `figures/confusion/plot_confusion_<MODEL>.py` | One script per model, counts baked in: 0, 6 and 10 dB × (SSL 20/10/5 %, supervised 100 %) |

## Models

| Name | Type | Input | Params |
|---|---|---|---|
| `Bi_LSTM` | traditional | I/Q | 536,600 |
| `PI_Bi_LSTM` | physics-informed | I/Q + A + φ | 538,648 |
| `PETCGDNN` | traditional | I/Q (phase-corrected) | 137,932 |
| `PI_Bi_PETCGDNN` | physics-informed | I'/Q' + A' + φ' after phase correction | 139,132 |
| `CNN` | traditional | I/Q, widths {256,128,64,64} | 1,777,304 |
| `PI_CNN` | physics-informed | I/Q + A + φ, widths {192,96,48,48} | 1,985,048 |
| `BiStream_Attn_CNNLSTM` | recent method [Kumar et al., 2024] | I/Q (A, φ internal) | 170,777 |
| `CANRN` | recent method [Bai et al., 2023] | I/Q | 8,486,424 |
| `CSPMNet` | recent method [Zhang et al., 2026] | I/Q | 8,453,465 |

All models take a raw `(B, 1024, 2)` I/Q batch; physics-informed variants
compute amplitude `A = sqrt(I² + Q² + 1e-8)` and phase `φ = atan2(Q, I)`
internally. The three recent methods are re-implementations from their
published descriptions, adapted to single-carrier 1024-sample frames.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Download RML2018.01A (`GOLD_XYZ_OSC.0001_1024.hdf5`, datasets `X`, `Y`, `Z`)
from DeepSig and point to it:

```bash
export HDF5_DATA_PATH=/path/to/GOLD_XYZ_OSC.0001_1024.hdf5
```

## Usage

```bash
# full experiment: all nine models, supervised + self-training
python train.py

# subsets
python train.py --models PI_Bi_LSTM Bi_LSTM --modes ssl --ratios 0.05
python train.py --nodes 0 6 10 --thresholds 0.70

# one model per script (self-contained, same settings and output layout)
python per_model/train_PI_Bi_LSTM.py
python per_model/train_CANRN.py --modes ssl --ratios 0.10 0.05

# quick smoke test (tiny data, 2 epochs)
python train.py --quick --models PI_Bi_LSTM

# results -> CSV tables
python summarize_results.py --root results --out results_summary

# Table I (run on the CPU you report; use one thread for stable timing)
python benchmark_complexity.py --threads 1

# paper figures (values baked into the scripts)
python figures/plot_accuracy_vs_snr.py
python figures/plot_confusion.py

# per-model confusion matrices (baked counts)
python figures/confusion/plot_confusion_PI_Bi_LSTM.py            # 0/6/10 dB in one figure
python figures/confusion/plot_confusion_PI_Bi_LSTM.py --split    # one figure per SNR
python figures/confusion/plot_confusion_CANRN.py --dump-csv      # also write CSV counts

# confusion matrices from your own train.py results
python figures/plot_confusion_per_model.py --root results --snrs 0 6 10
```

Lower `--batch-size` (default 400, as in the paper) if training runs out of memory.

### Confusion-matrix coverage

The baked per-model scripts hold the matrices saved by the original runs.
Those runs stored per-SNR confusion matrices only at 0 and 6 dB, except for
Bi-LSTM and PI-Bi-LSTM, which also have 10 dB. `train.py` saves
a matrix at every SNR, so a re-run fills the remaining 10 dB panels through
`plot_confusion_per_model.py`. Self-training panels show the best-scoring stage
at node z* (the rule used for the paper's confusion figure).

## Protocol

| Setting | Value |
|---|---|
| Split | 80 % train / 20 % test (fixed seed 2024) |
| Labeled fractions | supervised 100/20/10/5 %, self-training 20/10/5 % |
| Stage 1 (seed) | Adam, lr 1e-3, batch 400, ≤150 epochs, early stopping patience 15, lr ×0.5 after 5 stagnant epochs (min 1e-7) |
| Stage 2 (self-training) | unlabeled pool restricted to {z*−2, z*, z*+2} dB, K = 5 chunks, confidence threshold τ swept over {0.60, 0.65, 0.70, 0.75, 0.80}, ≤50 epochs per iteration |
| Pseudo-labels | max softmax ≥ τ, accumulated across iterations, equal weight to labeled samples |
| Evaluation | per SNR node, −14 to 28 dB in 2 dB steps |

Results are written as JSON under `results/`; `summarize_results.py` reports
each node's seed accuracy and the mean and standard deviation across
thresholds.

## Citation

```bibtex
@article{mahabub2026piamc,
  author  = {Mahabub, Atik and Vakili, Shervin},
  title   = {Physics-Informed Representations for Semi-Supervised Automatic
             Modulation Classification},
  journal = {IEEE Signal Processing Letters},
  note    = {submitted},
  year    = {2026}
}
```

## License

Released under the MIT License; see [`LICENSE`](LICENSE). The RML2018.01A
dataset is distributed by DeepSig under its own terms and is not included.

## Acknowledgement

Supported by the Natural Sciences and Engineering Research Council of Canada
(NSERC), Grant RGPIN 2024-06358, and Calcul Québec.
