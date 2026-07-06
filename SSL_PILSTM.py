import os
import time
import json
import h5py
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------- Configuration -----------------
HDF5_FILE_PATH = os.environ.get('HDF5_DATA_PATH', 'RML2018.hdf5')

# Dedicated Output Folders
RESULTS_DIR_SOFTMAX    = 'results_SNRs_ssl_PILSTM_Softmax'
RESULTS_DIR_SUPERVISED = 'results_SNRs_Supervised_PILSTM'

for d in [RESULTS_DIR_SOFTMAX, RESULTS_DIR_SUPERVISED]:
    if not os.path.exists(d):
        os.makedirs(d)

# Training Hyperparameters
BASE_EPOCHS = 150
RETRAIN_EPOCHS = 50
BATCH_SIZE = 400
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-7

# Data Split Configuration
SSL_LABELED_RATIOS = [0.20, 0.10, 0.05]
SUPERVISED_RATIOS = [1.0, 0.20, 0.10, 0.05]

# SSL Loop Config
ITERATION_VARIANTS = [5]
CONFIDENCE_THRESHOLDS = np.arange(0.60, 0.81, 0.05)
MC_SAMPLES = 10  # Number of forward passes for MC Dropout
DROPOUT_RATES = np.arange(0.1, 0.51, 0.1) # Hyperparameter search grid

# Node Definitions: {Center_SNR: [Range]}
NODES_CONFIG = {
    -6: [-8, -6, -4],
    -4: [-6, -4, -2],
    -2: [-4, -2, 0],
     0: [-2, 0, 2],
     2: [ 0, 2, 4],
     4: [ 2, 4, 6],
     6: [4, 6, 8],
     8: [6, 8, 10],
    10: [8, 10, 12],
    12: [10, 12, 14],
    14: [12, 14, 16],
    16: [14, 16, 18],
    18: [16, 18, 20],
    20: [18, 20, 22],
    22: [20, 22, 24],
    24: [22, 24, 26],
    26: [24, 26, 28],
    28: [26, 28, 30],
}

CLASSES = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Helpers -----------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return super().default(obj)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)

def enable_dropout(model):
    """Function to force dropout layers to remain active during test-time (for MC Dropout)"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# ----------------- Models -----------------
def get_physics_features(x):
    """Calculates Amplitude and Phase from I/Q data and appends them to create a 4D feature set."""
    i = x[:, :, 0]
    q = x[:, :, 1]
    amp = torch.sqrt(i**2 + q**2 + 1e-8)
    phase = torch.atan2(q, i)
    return torch.stack((i, q, amp, phase), dim=-1)

class PI_Bi_LSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, num_classes=24, dr=0.5):
        super(PI_Bi_LSTM, self).__init__()
        # input_dim increased to 4. bidirectional set to True.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dr if num_layers > 1 else 0.0, bidirectional=True)
        self.dropout = nn.Dropout(dr)
        self.fc = nn.Linear(hidden_dim * 2, num_classes) # * 2 for bidirectional
        
    def forward(self, x):
        # Apply Physics logic dynamically
        x_pi = get_physics_features(x)
        lstm_out, (h_n, c_n) = self.lstm(x_pi)
        
        # Concat last hidden state from forward [-2] and backward [-1] directions
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ----------------- Dataset -----------------
class RMLDataset(Dataset):
    def __init__(self, X_data, Y_data, Z_data, indices):
        self.X_data = X_data
        self.Y_data = Y_data
        self.Z_data = Z_data
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, i):
        idx = self.indices[i]
        x_sample = self.X_data[idx]
        y_raw = self.Y_data[idx]
        if isinstance(y_raw, (np.ndarray, list, torch.Tensor)) and len(y_raw.shape) > 0 and y_raw.shape[0] > 1:
             y_tensor = torch.tensor(np.argmax(y_raw), dtype=torch.long)
        else:
             y_tensor = torch.tensor(y_raw, dtype=torch.long)

        data_output = torch.tensor(x_sample, dtype=torch.float32)

        if self.Z_data is not None:
             z_val = self.Z_data[idx]
             return data_output, y_tensor, torch.tensor(z_val, dtype=torch.float32)
        
        return data_output, y_tensor

class PseudoLabelDataset(Dataset):
    def __init__(self, X_data, Pseudo_Y):
        self.X_data = X_data
        self.Y_data = Pseudo_Y

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, i):
        x_sample = self.X_data[i]
        y_tensor = torch.tensor(self.Y_data[i], dtype=torch.long)
        data_output = torch.tensor(x_sample, dtype=torch.float32)
            
        return data_output, y_tensor

# ----------------- Functions -----------------
def train_model(model, train_loader, val_loader, epochs, model_path, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR, min_lr=MIN_LR
    )
    best_loss = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'time_per_epoch': []}
    
    model.to(device)
    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            if len(batch) == 2: data, target = batch
            else: data, target, _ = batch
            
            if isinstance(data, (list, tuple)):
                data = tuple(d.to(device) for d in data)
                out = model(*data)
            else:
                out = model(data.to(device))
                
            target = target.to(device)
            loss = criterion(out, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            r_loss += loss.item() * target.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            
        t_loss = r_loss / total
        t_acc = correct / total
        
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2: data, target = batch
                else: data, target, _ = batch
                
                if isinstance(data, (list, tuple)):
                    data = tuple(d.to(device) for d in data)
                    out = model(*data)
                else:
                    out = model(data.to(device))
                    
                target = target.to(device)
                loss = criterion(out, target)
                v_loss += loss.item() * target.size(0)
                _, preds = torch.max(out, 1)
                v_correct += (preds == target).sum().item()
                v_total += target.size(0)
                
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        duration = time.time() - t_start
        
        history['train_loss'].append(t_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(val_acc)
        history['time_per_epoch'].append(duration)
        
        scheduler.step(val_loss)
        print(f"  Ep {epoch}: Train Loss {t_loss:.4f} Acc {t_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | {duration:.1f}s")
        
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  Early stopping.")
                break
    return history

def evaluate_model(model, dataloader, model_name):
    model.eval()
    model.to(device)
    all_preds, all_targets, all_snrs = [], [], []
    inference_times = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3: data, target, snr = batch
            else: data, target = batch; snr = torch.zeros(target.size(0))

            if isinstance(data, (list, tuple)):
                data = tuple(d.to(device) for d in data)
                t0 = time.time(); out = model(*data); t1 = time.time()
            else:
                data = data.to(device)
                t0 = time.time(); out = model(data); t1 = time.time()

            inference_times.append((t1 - t0) / target.size(0))
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_snrs.extend(snr.cpu().numpy())

    avg_inf_time = np.mean(inference_times)
    unique_snrs = np.unique(all_snrs)
    unique_snrs.sort()
    snr_metrics = {}
    cm_per_snr = {}
    
    for s in unique_snrs:
        indices = np.where(np.array(all_snrs) == s)[0]
        y_true_s = np.array(all_targets)[indices]
        y_pred_s = np.array(all_preds)[indices]
        acc = accuracy_score(y_true_s, y_pred_s)
        cm = confusion_matrix(y_true_s, y_pred_s, labels=range(len(CLASSES)))
        snr_metrics[str(int(s))] = acc
        cm_per_snr[str(int(s))] = cm.tolist()
        
    overall_acc = accuracy_score(all_targets, all_preds)
    return {
        "model": model_name,
        "overall_accuracy": overall_acc,
        "avg_inference_time_sec": avg_inf_time,
        "snr_accuracies": snr_metrics,
        "cm_per_snr": cm_per_snr
    }

def get_pseudo_labels_detailed(model, dataloader, threshold=0.9, use_mc_dropout=True):
    """
    Detailed pseudo-labeling with Oracle Accuracy per SNR.
    Supports either MC Dropout or standard Softmax.
    """
    model.eval()
    if use_mc_dropout:
        enable_dropout(model) # Keep dropout active for MC sampling
        
    model.to(device)
    confident_indices, pseudo_labels = [], []
    total_samples = 0
    batch_start_idx = 0
    oracle_hits = [] 
    
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0]
            target_oracle = batch[1].to(device) if len(batch) >= 2 else None
            z_snr = batch[2].to(device) if len(batch) >= 3 else None
            
            if isinstance(data, (list, tuple)):
                data_dev = tuple(d.to(device) for d in data)
            else:
                data_dev = data.to(device)
                
            if use_mc_dropout:
                # MC Dropout Forward Passes
                mc_probs = []
                for _ in range(MC_SAMPLES):
                    if isinstance(data_dev, tuple):
                        logits = model(*data_dev)
                    else:
                        logits = model(data_dev)
                    mc_probs.append(F.softmax(logits, dim=1))
                
                mc_probs = torch.stack(mc_probs)
                probs = mc_probs.mean(dim=0)
            else:
                # Standard Softmax Single Pass
                if isinstance(data_dev, tuple):
                    logits = model(*data_dev)
                else:
                    logits = model(data_dev)
                probs = F.softmax(logits, dim=1)
            
            max_probs, preds = torch.max(probs, 1)
            mask = max_probs >= threshold
            
            if target_oracle is not None and mask.sum() > 0:
                matches = (preds[mask] == target_oracle[mask]).cpu().numpy().flatten()
                snrs = z_snr[mask].cpu().numpy().flatten() if z_snr is not None else np.zeros(len(matches))
                for match, s in zip(matches, snrs):
                    oracle_hits.append((bool(match), float(s)))

            local_hits = torch.nonzero(mask).squeeze(1).cpu().numpy()
            global_hits = local_hits + batch_start_idx
            
            confident_indices.extend(global_hits.tolist())
            pseudo_labels.extend(preds[mask].cpu().numpy().tolist())
            batch_start_idx += probs.size(0); total_samples += probs.size(0)
    
    if len(oracle_hits) > 0:
        oracle_hits_arr = np.array(oracle_hits)
        correct_flags = oracle_hits_arr[:, 0]
        snr_vals = oracle_hits_arr[:, 1]
        overall_oracle_acc = np.mean(correct_flags)
        unique_s = np.unique(snr_vals)
        unique_s.sort()
        snr_oracle_acc = {}
        for s in unique_s:
            s_mask = (snr_vals == s)
            acc_s = np.mean(correct_flags[s_mask])
            snr_oracle_acc[str(int(s))] = float(acc_s)
    else:
        overall_oracle_acc = 0.0
        snr_oracle_acc = {}

    return confident_indices, pseudo_labels, total_samples, overall_oracle_acc, snr_oracle_acc

# ----------------- Pipeline 1: SSL -----------------
def run_ssl_pipeline(X_all, Y_all, Z_all, indices, results_dir, use_mc_dropout):
    mode_str = "MC DROPOUT" if use_mc_dropout else "SOFTMAX ONLY"
    
    for ratio in SSL_LABELED_RATIOS:
        percentage_str = int(ratio * 100)
        print(f"\n" + "="*70)
        print(f" STARTING {mode_str} EXPERIMENT FOR SSL RATIO: {percentage_str}%")
        print("="*70)

        ratio_dir = os.path.join(results_dir, f'ssl_{percentage_str}_percent')
        if not os.path.exists(ratio_dir): os.makedirs(ratio_dir)

        n_labeled = int(len(indices) * ratio)
        idx_labeled_total = indices[:n_labeled]
        idx_unlabeled_pool = indices[n_labeled:]
        
        n_lab_train = int(len(idx_labeled_total) * 0.8)
        idx_base_train = idx_labeled_total[:n_lab_train]
        idx_base_val = idx_labeled_total[n_lab_train:]
        
        print(f"Global Data Split:")
        print(f"  Base Labeled Train: {len(idx_base_train)}")
        print(f"  Base Labeled Val:   {len(idx_base_val)}")
        print(f"  Unlabeled Pool:     {len(idx_unlabeled_pool)}")
        
        model_name = 'PI_Bi_LSTM'
        model_dir = os.path.join(ratio_dir, 'models', model_name)
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        pth_dir = os.path.join(model_dir, 'pth')
        if not os.path.exists(pth_dir): os.makedirs(pth_dir)

        ds_base_train = RMLDataset(X_all, Y_all, None, idx_base_train)
        ds_base_val = RMLDataset(X_all, Y_all, None, idx_base_val)
        dl_base_train = DataLoader(ds_base_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_base_val = DataLoader(ds_base_val, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"--- Phase 1: Base Model (Hyperparameter Search for Dropout) ---")
        best_dr = 0.5
        best_val_loss = float('inf')
        best_base_hist = None
        base_path = os.path.join(pth_dir, 'base_model.pth')
        
        for dr in DROPOUT_RATES:
            dr = round(dr, 1)
            print(f"\n  Testing Dropout Rate: {dr}")
            temp_model = PI_Bi_LSTM(num_classes=len(CLASSES), dr=dr)
            temp_path = os.path.join(pth_dir, f'temp_base_dr_{dr}.pth')
            
            hist = train_model(temp_model, dl_base_train, dl_base_val, 
                               BASE_EPOCHS, temp_path, patience=EARLY_STOPPING_PATIENCE)
            
            min_val_loss = min(hist['val_loss'])
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_dr = dr
                best_base_hist = hist
                shutil.copy(temp_path, base_path)
                
        print(f"\n--- Selected Best Dropout Rate: {best_dr} with Val Loss: {best_val_loss:.4f} ---")
        
        base_model = PI_Bi_LSTM(num_classes=len(CLASSES), dr=best_dr)
        base_model.load_state_dict(torch.load(base_path))
        save_json(best_base_hist, os.path.join(model_dir, 'base_history.json'))

        for node_center, node_range in NODES_CONFIG.items():
            node_name = f"Node_{node_center}dB"
            print(f"\n>>> Processing Node: {node_center}dB (Range: {node_range})")
            node_dir = os.path.join(model_dir, node_name)
            if not os.path.exists(node_dir): os.makedirs(node_dir)

            pool_snrs = Z_all[idx_unlabeled_pool].flatten()
            node_mask = np.isin(pool_snrs, node_range)
            filtered_pool_indices = idx_unlabeled_pool[node_mask]
            
            np.random.shuffle(filtered_pool_indices)
            n_node_test = int(len(filtered_pool_indices) * 0.2)
            idx_test_node = filtered_pool_indices[:n_node_test]
            idx_teaching_pool = filtered_pool_indices[n_node_test:]
            
            ds_test_node = RMLDataset(X_all, Y_all, Z_all, idx_test_node)
            dl_test_node = DataLoader(ds_test_node, batch_size=BATCH_SIZE, shuffle=False)
            
            base_node_metrics = evaluate_model(base_model, dl_test_node, f"{model_name}_Base_{node_name}")
            save_json(base_node_metrics, os.path.join(node_dir, 'base_metrics.json'))

            for n_chunks in ITERATION_VARIANTS:
                chunk_dir = os.path.join(node_dir, f'chunks_{n_chunks}')
                if not os.path.exists(chunk_dir): os.makedirs(chunk_dir)

                curr_pool_indices = idx_teaching_pool.copy()
                np.random.shuffle(curr_pool_indices)
                if len(curr_pool_indices) < n_chunks:
                    continue
                    
                chunks = np.array_split(curr_pool_indices, n_chunks)
                
                for threshold in CONFIDENCE_THRESHOLDS:
                    thresh_str = f"{threshold:.2f}"
                    current_model = PI_Bi_LSTM(num_classes=len(CLASSES), dr=best_dr)
                    current_model.load_state_dict(torch.load(base_path))
                    
                    accumulated_pseudo_ds = None
                    chunk_results = []
                    
                    for i in range(n_chunks):
                        iter_num = i + 1
                        chunk_idx = chunks[i]
                        ds_chunk = RMLDataset(X_all, Y_all, Z_all, chunk_idx)
                        dl_chunk = DataLoader(ds_chunk, batch_size=BATCH_SIZE, shuffle=False)
                        
                        conf_idx, pseudo_labels, chunk_size, oracle_acc, snr_oracle = get_pseudo_labels_detailed(
                            current_model, dl_chunk, threshold, use_mc_dropout=use_mc_dropout)
                        
                        stats = {
                            "node": node_name, "config_chunks": n_chunks,
                            "threshold": threshold, "iteration": iter_num,
                            "total_pool_size": chunk_size,
                            "pseudo_labeled_count": len(conf_idx),
                            "pseudo_labeled_percentage": (len(conf_idx)/chunk_size)*100 if chunk_size>0 else 0,
                            "oracle_accuracy_overall": oracle_acc,
                            "oracle_accuracy_per_snr": snr_oracle
                        }
                        
                        if len(conf_idx) > 0:
                            real_global_indices = [chunk_idx[loc] for loc in conf_idx]
                            X_new = X_all[real_global_indices]
                            ds_new = PseudoLabelDataset(X_new, pseudo_labels)
                            
                            if accumulated_pseudo_ds is None:
                                accumulated_pseudo_ds = ds_new
                            else:
                                accumulated_pseudo_ds = ConcatDataset([accumulated_pseudo_ds, ds_new])
                        
                        if accumulated_pseudo_ds is not None:
                            ds_combined = ConcatDataset([ds_base_train, accumulated_pseudo_ds])
                            dl_combined = DataLoader(ds_combined, batch_size=BATCH_SIZE, shuffle=True)
                            
                            ckpt_name = f'ssl_{node_name}_c{n_chunks}_t{thresh_str}_i{iter_num}.pth'
                            retrain_path = os.path.join(pth_dir, ckpt_name)
                            
                            hist_ssl = train_model(current_model, dl_combined, dl_base_val, 
                                                   RETRAIN_EPOCHS, retrain_path, patience=5)
                            
                            current_model.load_state_dict(torch.load(retrain_path))
                            metrics = evaluate_model(current_model, dl_test_node, f"SSL_{node_name}_Iter{iter_num}")
                            chunk_results.append({"stats": stats, "metrics": metrics, "history": hist_ssl})
                        else:
                            chunk_results.append({"stats": stats, "metrics": None, "history": None})

                    threshold_json_path = os.path.join(chunk_dir, f'threshold_{thresh_str}.json')
                    experiment_data = {
                        "node": node_name, "ratio_percent": percentage_str,
                        "chunks_config": n_chunks, "threshold": threshold,
                        "iterations_data": chunk_results
                    }
                    save_json(experiment_data, threshold_json_path)

        print(f"Finished {model_name} for {percentage_str}% {mode_str} Split.")

# ----------------- Pipeline 2: Fully Supervised Baselines -----------------
def run_supervised_pipeline(X_all, Y_all, Z_all, indices):
    # Extract Global Consistent Test Set (20% of entire dataset)
    n_test = int(0.20 * len(indices))
    idx_global_test = indices[:n_test]
    idx_train_pool = indices[n_test:]
    
    ds_test = RMLDataset(X_all, Y_all, Z_all, idx_global_test)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    
    model_name = 'PI_Bi_LSTM'

    for ratio in SUPERVISED_RATIOS:
        percentage_str = int(ratio * 100)
        print(f"\n" + "="*70)
        print(f" STARTING FULLY SUPERVISED EXPERIMENT FOR RATIO: {percentage_str}% of Training Data")
        print("="*70)

        ratio_dir = os.path.join(RESULTS_DIR_SUPERVISED, f'supervised_{percentage_str}_percent')
        if not os.path.exists(ratio_dir): os.makedirs(ratio_dir)

        # Truncate training pool to match requested ratio
        n_train_target = int(len(idx_train_pool) * ratio)
        current_train_indices = idx_train_pool[:n_train_target]
        
        n_lab_train = int(len(current_train_indices) * 0.8)
        idx_base_train = current_train_indices[:n_lab_train]
        idx_base_val = current_train_indices[n_lab_train:]
        
        print(f"Data Split for {percentage_str}%:")
        print(f"  Train:       {len(idx_base_train)}")
        print(f"  Validation:  {len(idx_base_val)}")
        print(f"  Global Test: {len(idx_global_test)}")
        
        model_dir = os.path.join(ratio_dir, 'models', model_name)
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        pth_dir = os.path.join(model_dir, 'pth')
        if not os.path.exists(pth_dir): os.makedirs(pth_dir)

        ds_base_train = RMLDataset(X_all, Y_all, None, idx_base_train)
        ds_base_val = RMLDataset(X_all, Y_all, None, idx_base_val)
        dl_base_train = DataLoader(ds_base_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_base_val = DataLoader(ds_base_val, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"--- Base Model (Hyperparameter Search for Dropout) ---")
        best_dr, best_val_loss, best_base_hist = 0.5, float('inf'), None
        base_path = os.path.join(pth_dir, 'base_model.pth')
        
        for dr in DROPOUT_RATES:
            dr = round(dr, 1)
            print(f"  Testing Dropout Rate: {dr}")
            temp_model = PI_Bi_LSTM(num_classes=len(CLASSES), dr=dr)
            temp_path = os.path.join(pth_dir, f'temp_base_dr_{dr}.pth')
            hist = train_model(temp_model, dl_base_train, dl_base_val, BASE_EPOCHS, temp_path, patience=EARLY_STOPPING_PATIENCE)
            
            min_val_loss = min(hist['val_loss'])
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss; best_dr = dr; best_base_hist = hist
                shutil.copy(temp_path, base_path)
                
        print(f"\n--- Selected Best Dropout Rate: {best_dr} with Val Loss: {best_val_loss:.4f} ---")
        base_model = PI_Bi_LSTM(num_classes=len(CLASSES), dr=best_dr)
        base_model.load_state_dict(torch.load(base_path))
        
        # Save History
        save_json(best_base_hist, os.path.join(model_dir, 'base_history.json'))

        # Evaluate on the Global Consistent Test set to compare directly against SSL results
        print(f"\n--- Evaluating Supervised {percentage_str}% on Global Test Set ---")
        metrics = evaluate_model(base_model, dl_test, f"{model_name}_Supervised_{percentage_str}pct")
        save_json(metrics, os.path.join(model_dir, 'test_metrics.json'))
        
        print(f"Overall Test Accuracy: {metrics['overall_accuracy']:.4f}")

# ----------------- Runner -----------------
if __name__ == '__main__':
    print("Loading Dataset...")
    try:
        h5 = h5py.File(HDF5_FILE_PATH, 'r')
        X_all = h5['X'][:]
        Y_all = h5['Y'][:]
        Z_all = h5['Z'][:] 
        h5.close()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    print(f"Using Full Dataset Range.")
    n_samples = X_all.shape[0]
    indices = np.arange(n_samples)
    np.random.seed(2024)
    np.random.shuffle(indices)

    # 1. Run Fully Supervised Baselines
    print("\n\n" + "="*80)
    print(">>> RUNNING EXPERIMENT 1: FULLY SUPERVISED BASELINES")
    print("="*80)
    run_supervised_pipeline(X_all, Y_all, Z_all, indices)

    # 2. Run SSL with Standard Softmax
    print("\n\n" + "="*80)
    print(">>> RUNNING EXPERIMENT 3: SSL WITH SOFTMAX ONLY")
    print("="*80)
    run_ssl_pipeline(X_all, Y_all, Z_all, indices, RESULTS_DIR_SOFTMAX, use_mc_dropout=False)

    print("\nAll experiments completed successfully!")