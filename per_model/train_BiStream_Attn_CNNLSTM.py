#!/usr/bin/env python3
"""BiStream-Attn-CNN-LSTM (recent method, Kumar et al., IEEE Commun. Lett., 2024) on RML2018.01A: supervised baselines and
SNR-localized self-training, in one self-contained file.

Same pipeline and settings as train.py, restricted to BiStream_Attn_CNNLSTM:
  supervised  one model per labeled fraction (100/20/10/5 %), evaluated on a
              fixed 20 % global test split
  ssl         a supervised seed per labeled fraction (20/10/5 %), then
              confidence-based self-training per SNR node z*, with the
              unlabeled pool restricted to {z*-2, z*, z*+2} dB, K = 5 chunks
              and the confidence threshold swept over the grid below
A 24x24 confusion matrix is saved for every SNR in each test set.

Results (JSON), in the layout read by summarize_results.py and the figure scripts:
  results/BiStream_Attn_CNNLSTM/Supervised/supervised_<pct>_percent/models/BiStream_Attn_CNNLSTM/test_metrics.json
  results/BiStream_Attn_CNNLSTM/SSL_Softmax/ssl_<pct>_percent/models/BiStream_Attn_CNNLSTM/Node_<z>dB/
        base_metrics.json
        chunks_5/threshold_<tau>.json

USAGE
  export HDF5_DATA_PATH=/path/to/GOLD_XYZ_OSC.0001_1024.hdf5
  python train_BiStream_Attn_CNNLSTM.py                       # full run, paper settings
  python train_BiStream_Attn_CNNLSTM.py --modes ssl --ratios 0.05
  python train_BiStream_Attn_CNNLSTM.py --nodes 0 6 10        # selected SNR nodes only
  python train_BiStream_Attn_CNNLSTM.py --quick               # smoke test (tiny run)
"""
import os
import json
import time
import argparse
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix, accuracy_score


# ----------------- Configuration (paper values) -----------------
BASE_EPOCHS = 150
RETRAIN_EPOCHS = 50
BATCH_SIZE = 400
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15     # Stage 1 (supervised seed)
RETRAIN_PATIENCE = 5             # Stage 2 (self-training fine-tuning)
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-7
DROPOUT_RATE = 0.5

SSL_LABELED_RATIOS = [0.20, 0.10, 0.05]
SUPERVISED_RATIOS = [1.0, 0.20, 0.10, 0.05]
N_CHUNKS = 5                                        # K
CONFIDENCE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
SEED = 2024

# node centre z* -> SNR levels in its local pool
NODES_CONFIG = {z: [z - 2, z, z + 2] for z in range(-14, 29, 2)}

CLASSES = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Model -----------------
class BiStream_Attn_CNNLSTM(nn.Module):
    def __init__(self, num_classes=24, dr=0.5):
        super(BiStream_Attn_CNNLSTM, self).__init__()
        # Stream 1: I/Q Data Processing
        self.conv_iq = nn.Conv1d(2, 64, kernel_size=5, padding=2)
        self.lstm_iq = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        
        # Stream 2: Amplitude/Phase Processing
        self.conv_ap = nn.Conv1d(2, 64, kernel_size=5, padding=2)
        self.lstm_ap = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        
        # Attention Mechanism (acting on concatenated features)
        self.attention = nn.Linear(256, 1) # 128 from IQ + 128 from AP
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(dr)

    def forward(self, x):
        # x: [B, 1024, 2] -> permute for CNN: [B, 2, 1024]
        iq_stream = x.permute(0, 2, 1) 
        
        # Create Amplitude/Phase stream computationally
        amp = torch.sqrt(iq_stream[:, 0:1, :]**2 + iq_stream[:, 1:2, :]**2 + 1e-8)
        phase = torch.atan2(iq_stream[:, 1:2, :], iq_stream[:, 0:1, :])
        ap_stream = torch.cat([amp, phase], dim=1) 

        # CNN Feature Extraction
        f_iq = F.relu(self.conv_iq(iq_stream)).permute(0, 2, 1) # Back to [B, 1024, 64] for LSTM
        f_ap = F.relu(self.conv_ap(ap_stream)).permute(0, 2, 1)

        # LSTM Sequence Modeling
        out_iq, _ = self.lstm_iq(f_iq) # [B, 1024, 128]
        out_ap, _ = self.lstm_ap(f_ap) # [B, 1024, 128]

        # Bi-Stream Fusion
        combined = torch.cat([out_iq, out_ap], dim=-1) # [B, 1024, 256]

        # Attention Weighting
        attn_weights = F.softmax(self.attention(combined), dim=1) # [B, 1024, 1]
        attended_features = torch.sum(combined * attn_weights, dim=1) # [B, 256]

        out = self.drop(F.relu(self.fc1(attended_features)))
        return self.fc2(out)


# ----------------- Helpers -----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location=device))


class RMLDataset(Dataset):
    def __init__(self, X_data, Y_data, Z_data, indices):
        self.X_data, self.Y_data, self.Z_data = X_data, Y_data, Z_data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        y_raw = self.Y_data[idx]
        if np.ndim(y_raw) > 0 and np.shape(y_raw)[0] > 1:      # one-hot labels
            y = torch.tensor(np.argmax(y_raw), dtype=torch.long)
        else:
            y = torch.tensor(y_raw, dtype=torch.long)
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)
        if self.Z_data is not None:
            return x, y, torch.tensor(float(np.ravel(self.Z_data[idx])[0]), dtype=torch.float32)
        return x, y


class PseudoLabelDataset(Dataset):
    def __init__(self, X_data, pseudo_y):
        self.X_data, self.Y_data = X_data, pseudo_y

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, i):
        return (torch.tensor(self.X_data[i], dtype=torch.float32),
                torch.tensor(self.Y_data[i], dtype=torch.long))


# ----------------- Training / evaluation -----------------
def train_model(model, train_loader, val_loader, epochs, model_path, patience):
    """Adam + ReduceLROnPlateau, early stopping on validation loss.
    Labeled and pseudo-labeled samples are weighted equally."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR, min_lr=MIN_LR)
    best_loss, no_improve = float('inf'), 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'time_per_epoch': []}
    model.to(device)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            out = model(data)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * target.size(0)
            correct += (out.argmax(1) == target).sum().item()
            total += target.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                data, target = batch[0].to(device), batch[1].to(device)
                out = model(data)
                v_loss += criterion(out, target).item() * target.size(0)
                v_correct += (out.argmax(1) == target).sum().item()
                v_total += target.size(0)

        t_loss, val_loss = r_loss / total, v_loss / v_total
        history['train_loss'].append(t_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(correct / total)
        history['val_acc'].append(v_correct / v_total)
        history['time_per_epoch'].append(time.time() - t0)
        scheduler.step(val_loss)
        print(f"  Ep {epoch}: train {t_loss:.4f}/{correct/total:.4f} | "
              f"val {val_loss:.4f}/{v_correct/v_total:.4f}")

        if val_loss < best_loss:
            best_loss, no_improve = val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  Early stopping.")
                break
    return history


def evaluate_model(model, dataloader, model_name):
    """Overall and per-SNR accuracy, plus a 24x24 confusion matrix per SNR."""
    model.eval()
    model.to(device)
    preds_all, targets_all, snrs_all, times = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            data, target = batch[0].to(device), batch[1]
            snr = batch[2] if len(batch) == 3 else torch.zeros(target.size(0))
            t0 = time.time()
            out = model(data)
            times.append((time.time() - t0) / target.size(0))
            preds_all.extend(out.argmax(1).cpu().numpy())
            targets_all.extend(target.numpy())
            snrs_all.extend(snr.numpy().reshape(-1))

    preds_all, targets_all, snrs_all = map(np.array, (preds_all, targets_all, snrs_all))
    snr_acc, cm_per_snr = {}, {}
    for s in np.sort(np.unique(snrs_all)):
        m = snrs_all == s
        snr_acc[str(int(s))] = accuracy_score(targets_all[m], preds_all[m])
        cm_per_snr[str(int(s))] = confusion_matrix(
            targets_all[m], preds_all[m], labels=range(len(CLASSES))).tolist()
    return {"model": model_name,
            "overall_accuracy": accuracy_score(targets_all, preds_all),
            "avg_inference_time_sec": float(np.mean(times)),
            "snr_accuracies": snr_acc,
            "cm_per_snr": cm_per_snr}


def get_pseudo_labels(model, dataloader, threshold):
    """Indices (within the loader) and labels of predictions with max softmax
    >= threshold, plus the oracle accuracy of the accepted pseudo-labels."""
    model.eval()
    model.to(device)
    conf_idx, labels, hits = [], [], []
    start, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            probs = F.softmax(model(batch[0].to(device)), dim=1)
            max_p, preds = probs.max(1)
            mask = (max_p >= threshold).cpu()
            preds = preds.cpu()
            if len(batch) >= 2 and mask.any():
                hits.extend((preds[mask] == batch[1][mask]).numpy().tolist())
            local = torch.nonzero(mask).squeeze(1).numpy()
            conf_idx.extend((local + start).tolist())
            labels.extend(preds[mask].numpy().tolist())
            start += probs.size(0)
            total += probs.size(0)
    oracle = float(np.mean(hits)) if hits else 0.0
    return conf_idx, labels, total, oracle


# ----------------- Pipeline 1: supervised baselines -----------------
def run_supervised(X, Y, Z, indices, out_dir, model_name, ratios):
    n_test = int(0.20 * len(indices))
    idx_test, idx_pool = indices[:n_test], indices[n_test:]
    dl_test = DataLoader(RMLDataset(X, Y, Z, idx_test), batch_size=BATCH_SIZE)

    for ratio in ratios:
        pct = int(round(ratio * 100))
        print(f"\n=== {model_name}: supervised {pct}% ===")
        idx_train = idx_pool[:int(len(idx_pool) * ratio)]
        n_tr = int(len(idx_train) * 0.8)
        dl_tr = DataLoader(RMLDataset(X, Y, None, idx_train[:n_tr]),
                           batch_size=BATCH_SIZE, shuffle=True)
        dl_va = DataLoader(RMLDataset(X, Y, None, idx_train[n_tr:]),
                           batch_size=BATCH_SIZE)

        model_dir = os.path.join(out_dir, f'supervised_{pct}_percent', 'models', model_name)
        os.makedirs(os.path.join(model_dir, 'pth'), exist_ok=True)
        ckpt = os.path.join(model_dir, 'pth', 'base_model.pth')

        model = build_model(len(CLASSES), DROPOUT_RATE)
        hist = train_model(model, dl_tr, dl_va, BASE_EPOCHS, ckpt, EARLY_STOPPING_PATIENCE)
        load_state(model, ckpt)
        save_json(hist, os.path.join(model_dir, 'base_history.json'))
        metrics = evaluate_model(model, dl_test, f"{model_name}_Supervised_{pct}pct")
        save_json(metrics, os.path.join(model_dir, 'test_metrics.json'))
        print(f"  test accuracy {metrics['overall_accuracy']:.4f}")


# ----------------- Pipeline 2: SNR-localized self-training -----------------
def run_ssl(X, Y, Z, indices, out_dir, model_name, ratios, thresholds, nodes):
    for ratio in ratios:
        pct = int(round(ratio * 100))
        print(f"\n=== {model_name}: self-training {pct}% labeled ===")
        n_lab = int(len(indices) * ratio)
        idx_lab, idx_pool = indices[:n_lab], indices[n_lab:]
        n_tr = int(len(idx_lab) * 0.8)
        ds_tr = RMLDataset(X, Y, None, idx_lab[:n_tr])
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
        dl_va = DataLoader(RMLDataset(X, Y, None, idx_lab[n_tr:]), batch_size=BATCH_SIZE)

        model_dir = os.path.join(out_dir, f'ssl_{pct}_percent', 'models', model_name)
        pth_dir = os.path.join(model_dir, 'pth')
        os.makedirs(pth_dir, exist_ok=True)
        base_ckpt = os.path.join(pth_dir, 'base_model.pth')

        # Stage 1: supervised seed on the labeled set
        base = build_model(len(CLASSES), DROPOUT_RATE)
        hist = train_model(base, dl_tr, dl_va, BASE_EPOCHS, base_ckpt, EARLY_STOPPING_PATIENCE)
        load_state(base, base_ckpt)
        save_json(hist, os.path.join(model_dir, 'base_history.json'))

        pool_snrs = Z[idx_pool].flatten()
        for z_star in nodes:
            node_range = NODES_CONFIG[z_star]
            node = f"Node_{z_star}dB"
            node_dir = os.path.join(model_dir, node)
            os.makedirs(node_dir, exist_ok=True)
            print(f"\n>>> {node} (pool SNRs {node_range})")

            node_idx = idx_pool[np.isin(pool_snrs, node_range)].copy()
            np.random.shuffle(node_idx)
            n_test = int(len(node_idx) * 0.2)
            idx_test, idx_teach = node_idx[:n_test], node_idx[n_test:]
            dl_test = DataLoader(RMLDataset(X, Y, Z, idx_test), batch_size=BATCH_SIZE)
            save_json(evaluate_model(base, dl_test, f"{model_name}_Base_{node}"),
                      os.path.join(node_dir, 'base_metrics.json'))

            chunk_dir = os.path.join(node_dir, f'chunks_{N_CHUNKS}')
            os.makedirs(chunk_dir, exist_ok=True)
            pool = idx_teach.copy()
            np.random.shuffle(pool)
            if len(pool) < N_CHUNKS:
                continue
            chunks = np.array_split(pool, N_CHUNKS)

            # Stage 2: self-training, one run per confidence threshold
            for tau in thresholds:
                model = build_model(len(CLASSES), DROPOUT_RATE)
                load_state(model, base_ckpt)
                pseudo_ds, results = None, []
                for k, chunk in enumerate(chunks, start=1):
                    dl_chunk = DataLoader(RMLDataset(X, Y, Z, chunk), batch_size=BATCH_SIZE)
                    conf, labels, size, oracle = get_pseudo_labels(model, dl_chunk, tau)
                    stats = {"node": node, "threshold": tau, "iteration": k,
                             "total_pool_size": size, "pseudo_labeled_count": len(conf),
                             "pseudo_labeled_percentage": 100 * len(conf) / size if size else 0,
                             "oracle_accuracy_overall": oracle}
                    if conf:
                        new = PseudoLabelDataset(X[chunk[conf]], labels)
                        pseudo_ds = new if pseudo_ds is None else ConcatDataset([pseudo_ds, new])
                    if pseudo_ds is None:
                        results.append({"stats": stats, "metrics": None, "history": None})
                        continue
                    dl_comb = DataLoader(ConcatDataset([ds_tr, pseudo_ds]),
                                         batch_size=BATCH_SIZE, shuffle=True)
                    ckpt = os.path.join(pth_dir, f'ssl_{node}_c{N_CHUNKS}_t{tau:.2f}_i{k}.pth')
                    h = train_model(model, dl_comb, dl_va, RETRAIN_EPOCHS, ckpt, RETRAIN_PATIENCE)
                    load_state(model, ckpt)
                    m = evaluate_model(model, dl_test, f"SSL_{node}_Iter{k}")
                    results.append({"stats": stats, "metrics": m, "history": h})

                save_json({"node": node, "ratio_percent": f"{pct}%",
                           "chunks_config": N_CHUNKS, "threshold": tau,
                           "iterations_data": results},
                          os.path.join(chunk_dir, f'threshold_{tau:.2f}.json'))


# ----------------- Main -----------------
MODEL_NAME = "BiStream_Attn_CNNLSTM"


def build_model(num_classes=24, dr=0.5):
    return BiStream_Attn_CNNLSTM(num_classes=num_classes, dr=dr)


def main():
    global BASE_EPOCHS, RETRAIN_EPOCHS, BATCH_SIZE
    ap = argparse.ArgumentParser(description="BiStream-Attn-CNN-LSTM: supervised + SNR-localized self-training")
    ap.add_argument('--data', default=os.environ.get('HDF5_DATA_PATH', 'RML2018.hdf5'),
                    help='RML2018.01A HDF5 file with datasets X, Y, Z')
    ap.add_argument('--modes', nargs='*', default=['supervised', 'ssl'],
                    choices=['supervised', 'ssl'])
    ap.add_argument('--ratios', type=float, nargs='*', default=None,
                    help='labeled fractions (default: paper values per mode)')
    ap.add_argument('--thresholds', type=float, nargs='*', default=CONFIDENCE_THRESHOLDS)
    ap.add_argument('--nodes', type=int, nargs='*', default=list(NODES_CONFIG))
    ap.add_argument('--out', default='results')
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                    help='mini-batch size (paper: 400); lower it if memory is short')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 epochs, 30k frames, 2 nodes, 1 threshold')
    a = ap.parse_args()

    BATCH_SIZE = a.batch_size
    set_seed(a.seed)
    print(f"device: {device} | model: {MODEL_NAME} "
          f"({sum(p.numel() for p in build_model().parameters()):,} parameters)")
    with h5py.File(a.data, 'r') as h5:
        X, Y, Z = h5['X'][:], h5['Y'][:], h5['Z'][:]
    print(f"loaded {X.shape[0]} frames from {a.data}")

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    if a.quick:
        BASE_EPOCHS, RETRAIN_EPOCHS = 2, 1
        indices = indices[:30000]
        a.nodes, a.thresholds = a.nodes[:2], a.thresholds[:1]

    base_dir = os.path.join(a.out, MODEL_NAME)
    if 'supervised' in a.modes:
        run_supervised(X, Y, Z, indices, os.path.join(base_dir, 'Supervised'),
                       MODEL_NAME, a.ratios or SUPERVISED_RATIOS)
    if 'ssl' in a.modes:
        run_ssl(X, Y, Z, indices, os.path.join(base_dir, 'SSL_Softmax'),
                MODEL_NAME, a.ratios or SSL_LABELED_RATIOS, a.thresholds, a.nodes)
    print(f"\n{MODEL_NAME}: all runs finished.")


if __name__ == '__main__':
    main()
