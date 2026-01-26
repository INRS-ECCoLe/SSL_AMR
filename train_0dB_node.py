import os
import time
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------- Configuration -----------------
HDF5_FILE_PATH = os.environ.get('HDF5_DATA_PATH', 'RML2018.hdf5')
RESULTS_DIR = 'results_ssl_node_0dB'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

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
GLOBAL_TRAIN_SPLIT = 0.80   # 80% of Data for Training (Labeled + Pool)
GLOBAL_TEST_SPLIT = 0.20    # 20% of Data for Final Evaluation
SSL_LABELED_RATIO = 0.30    # 30% of the GLOBAL TRAIN is Labeled
SSL_UNLABELED_RATIO = 0.70  # 70% of the GLOBAL TRAIN is Unlabeled Pool

# SSL Loop Config
ITERATION_VARIANTS = [3, 5, 7] # The "cumulative iteration" counts
CONFIDENCE_THRESHOLDS = np.arange(0.4, 0.81, 0.05) 
RETRAIN_SNR_RANGE = [-2, 0, 2] # The 0dB Node Range for the POOL ONLY

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
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)

# ----------------- Models -----------------

class PyTorch_CNN(nn.Module):
    def __init__(self, num_classes=24):
        super(PyTorch_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(2, 8), padding='same')
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(2, 8), padding='same')
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(2, 8), padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(2, 8), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1).unsqueeze(1) 
        x = F.relu(self.conv1(x)); x = self.pool(x); x = self.dropout(x)
        x = F.relu(self.conv2(x)); x = self.pool(x); x = self.dropout(x)
        x = F.relu(self.conv3(x)); x = self.pool(x); x = self.dropout(x)
        x = F.relu(self.conv4(x)); x = self.pool(x); x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PyTorch_PETCGDNN(nn.Module):
    def __init__(self, num_classes=24, dr=0.5):
        super(PyTorch_PETCGDNN, self).__init__()
        self.phase_fc = nn.Linear(1024 * 2, 1)
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(8, 2), padding='valid')
        self.conv1_2 = nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(5, 1), padding='valid')
        self.gru = nn.GRU(input_size=25, hidden_size=128, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x_iq, x_i, x_q):
        x_flat = torch.flatten(x_iq, 1)
        phase_offset = self.phase_fc(x_flat)
        cos_val, sin_val = torch.cos(phase_offset), torch.sin(phase_offset)
        y1, y2 = (x_i * cos_val) + (x_q * sin_val), (x_q * cos_val) - (x_i * sin_val)
        x_corrected = torch.cat([y1.unsqueeze(-1), y2.unsqueeze(-1)], dim=2)
        x3 = x_corrected.unsqueeze(1)
        x3 = F.relu(self.conv1_1(x3))
        x3 = F.relu(self.conv1_2(x3))
        x4 = x3.squeeze(-1).permute(0, 2, 1)
        gru_out, h_n = self.gru(x4)
        return self.fc_out(h_n[-1])

class PyTorch_LSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, num_classes=24):
        super(PyTorch_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# ----------------- Dataset -----------------
class RMLDataset(Dataset):
    def __init__(self, X_data, Y_data, Z_data, indices, mode='SIMPLE'):
        self.X_data = X_data
        self.Y_data = Y_data
        self.Z_data = Z_data
        self.indices = indices
        self.mode = mode
        
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

        # Mode-specific preprocessing
        if self.mode == 'PETCGDNN':
            x1_data = np.expand_dims(x_sample, axis=0)
            x2_data = x_sample[:, 0]
            x3_data = x_sample[:, 1]
            data_output = (
                torch.tensor(x1_data, dtype=torch.float32),
                torch.tensor(x2_data, dtype=torch.float32),
                torch.tensor(x3_data, dtype=torch.float32)
            )
        else:
            # Simple handling for LSTM and CNN (N, 1024, 2)
            data_output = torch.tensor(x_sample, dtype=torch.float32)

        if self.Z_data is not None:
             z_val = self.Z_data[idx]
             return data_output, y_tensor, torch.tensor(z_val, dtype=torch.float32)
        
        return data_output, y_tensor

class PseudoLabelDataset(Dataset):
    def __init__(self, X_data, Pseudo_Y, mode='SIMPLE'):
        self.X_data = X_data
        self.Y_data = Pseudo_Y
        self.mode = mode

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, i):
        x_sample = self.X_data[i]
        y_tensor = torch.tensor(self.Y_data[i], dtype=torch.long)
        
        if self.mode == 'PETCGDNN':
            x1_data = np.expand_dims(x_sample, axis=0)
            x2_data = x_sample[:, 0]
            x3_data = x_sample[:, 1]
            data_output = (
                torch.tensor(x1_data, dtype=torch.float32),
                torch.tensor(x2_data, dtype=torch.float32),
                torch.tensor(x3_data, dtype=torch.float32)
            )
        else:
            data_output = torch.tensor(x_sample, dtype=torch.float32)
            
        return data_output, y_tensor

# ----------------- Functions -----------------

def train_model(model, train_loader, val_loader, epochs, model_path, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=LR_SCHEDULER_PATIENCE, 
        factor=LR_SCHEDULER_FACTOR, 
        min_lr=MIN_LR
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

def get_pseudo_labels_detailed(model, dataloader, threshold=0.9):
    """
    Detailed pseudo-labeling with Oracle Accuracy per SNR
    """
    model.eval()
    model.to(device)
    confident_indices, pseudo_labels = [], []
    total_samples = 0
    batch_start_idx = 0
    
    # Storage for Oracle Stats
    # List of (is_correct, snr_value) for every confident prediction
    oracle_hits = [] 
    
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0]
            # Need Z (SNR) and Y (Target) for Oracle Stats
            if len(batch) >= 2: target_oracle = batch[1].to(device)
            else: target_oracle = None
            if len(batch) >= 3: z_snr = batch[2].to(device)
            else: z_snr = None

            if isinstance(data, (list, tuple)):
                data = tuple(d.to(device) for d in data); logits = model(*data)
            else:
                logits = model(data.to(device))
            
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, 1)
            mask = max_probs >= threshold
            
            # Oracle Check
            if target_oracle is not None and mask.sum() > 0:
                matches = (preds[mask] == target_oracle[mask]).cpu().numpy().flatten() # Ensure 1D bool
                snrs = z_snr[mask].cpu().numpy().flatten() if z_snr is not None else np.zeros(len(matches))
                for match, s in zip(matches, snrs):
                    oracle_hits.append((bool(match), float(s))) # Cast to standard python types

            local_hits = torch.nonzero(mask).squeeze(1).cpu().numpy()
            global_hits = local_hits + batch_start_idx
            
            confident_indices.extend(global_hits.tolist())
            pseudo_labels.extend(preds[mask].cpu().numpy().tolist())
            
            batch_start_idx += logits.size(0); total_samples += logits.size(0)
    
    # Compile Oracle Stats
    if len(oracle_hits) > 0:
        oracle_hits_arr = np.array(oracle_hits) # col 0: correct(1)/wrong(0), col 1: snr
        correct_flags = oracle_hits_arr[:, 0]
        snr_vals = oracle_hits_arr[:, 1]
        
        overall_oracle_acc = np.mean(correct_flags)
        
        # Per SNR Oracle
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

# ----------------- Main Pipeline -----------------

def run_ssl_pipeline():
    print("Loading Dataset...")
    try:
        h5 = h5py.File(HDF5_FILE_PATH, 'r')
        X_all = h5['X'][:]
        Y_all = h5['Y'][:]
        Z_all = h5['Z'][:] 
        h5.close()
    except Exception as e:
        print(f"Error loading data: {e}"); return

    # --- Use Full Dataset (All SNRs from -20dB to 30dB) ---
    print(f"Using Full Dataset Range.")
    
    n_samples = X_all.shape[0]
    indices = np.arange(n_samples)
    
    np.random.seed(2024)
    np.random.shuffle(indices)
    
    # --- 1. Split Global Data ---
    # 80% Train (Labeled+Pool), 20% Test (Evaluation)
    n_train_global = int(n_samples * GLOBAL_TRAIN_SPLIT)
    idx_train_global = indices[:n_train_global]
    idx_test_final = indices[n_train_global:] 
    
    print(f"Global Split: Train_Global={len(idx_train_global)}, Test_Final={len(idx_test_final)}")
    
    # --- 2. Split Train_Global -> Labeled (30%) & Unlabeled Pool (70%) ---
    n_labeled = int(len(idx_train_global) * SSL_LABELED_RATIO)
    idx_labeled_total = idx_train_global[:n_labeled]
    idx_unlabeled_pool = idx_train_global[n_labeled:]
    
    # Further split Labeled into Train/Val for Base Training
    n_lab_train = int(len(idx_labeled_total) * 0.8)
    idx_base_train = idx_labeled_total[:n_lab_train]
    idx_base_val = idx_labeled_total[n_lab_train:]
    
    print(f"SSL Data Config:")
    print(f"  Base Labeled Train: {len(idx_base_train)}")
    print(f"  Base Labeled Val:   {len(idx_base_val)}")
    print(f"  Unlabeled Pool:     {len(idx_unlabeled_pool)}")
    
    # --- 3. FILTER POOL FOR NODE RANGE [-2, 0, 2] ---
    print(f"Filtering Unlabeled Pool for Node Range: {RETRAIN_SNR_RANGE}")
    # Get SNRs for the pool samples
    pool_snrs = Z_all[idx_unlabeled_pool].flatten()
    # Create mask for node range
    node_mask = np.isin(pool_snrs, RETRAIN_SNR_RANGE)
    # Apply mask to indices
    filtered_pool_indices = idx_unlabeled_pool[node_mask]
    
    print(f"  -> Original Pool Size: {len(idx_unlabeled_pool)}")
    print(f"  -> Filtered Pool Size (Node Range): {len(filtered_pool_indices)}")

    models_config = [
        ('LSTM', PyTorch_LSTM, 'LSTM_GRU'),
        ('PETCGDNN', PyTorch_PETCGDNN, 'PETCGDNN'),
        ('CNN', PyTorch_CNN, 'SIMPLE')
    ]

    for model_name, ModelClass, data_mode in models_config:
        print(f"\n================ Processing {model_name} ================")
        
        # L1 Directory: Model (Grouped under 'models' folder)
        model_dir = os.path.join(RESULTS_DIR, 'models', model_name)
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        
        # L1.5: Central PTH Directory
        pth_dir = os.path.join(model_dir, 'pth')
        if not os.path.exists(pth_dir): os.makedirs(pth_dir)

        # Datasets
        ds_base_train = RMLDataset(X_all, Y_all, None, idx_base_train, mode=data_mode)
        ds_base_val = RMLDataset(X_all, Y_all, None, idx_base_val, mode=data_mode)
        dl_base_train = DataLoader(ds_base_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_base_val = DataLoader(ds_base_val, batch_size=BATCH_SIZE, shuffle=False)
        
        ds_test_final = RMLDataset(X_all, Y_all, Z_all, idx_test_final, mode=data_mode)
        dl_test_final = DataLoader(ds_test_final, batch_size=BATCH_SIZE, shuffle=False)
        
        # ---------------------------------------------------------
        # PHASE 1: Base Model Training (30% Labeled)
        # ---------------------------------------------------------
        print(f"--- Phase 1: Base Model (30% Labeled) ---")
        base_model = ModelClass(num_classes=len(CLASSES))
        base_path = os.path.join(pth_dir, 'base_model.pth')
        
        hist_base = train_model(base_model, dl_base_train, dl_base_val, 
                                BASE_EPOCHS, base_path, patience=EARLY_STOPPING_PATIENCE)
        
        base_model.load_state_dict(torch.load(base_path))
        base_metrics = evaluate_model(base_model, dl_test_final, f"{model_name}_Base")
        save_json(base_metrics, os.path.join(model_dir, 'base_metrics.json'))
        save_json(hist_base, os.path.join(model_dir, 'base_history.json'))

        # ---------------------------------------------------------
        # PHASE 2: SSL Loop (Variable Iterations)
        # ---------------------------------------------------------
        
        # Outer Loop: Number of Chunks (Cumulative Iterations)
        for n_chunks in ITERATION_VARIANTS:
            print(f"\n--- Phase 2: SSL Config | Chunks/Iterations: {n_chunks} ---")
            
            # L2 Directory: Chunks
            chunk_dir = os.path.join(model_dir, f'chunks_{n_chunks}')
            if not os.path.exists(chunk_dir): os.makedirs(chunk_dir)

            # Use FILTERED pool indices
            curr_pool_indices = filtered_pool_indices.copy()
            np.random.shuffle(curr_pool_indices)
            chunks = np.array_split(curr_pool_indices, n_chunks)
            
            # Middle Loop: Threshold
            for threshold in CONFIDENCE_THRESHOLDS:
                thresh_str = f"{threshold:.2f}"
                
                # Reset Model to Base
                current_model = ModelClass(num_classes=len(CLASSES))
                current_model.load_state_dict(torch.load(base_path))
                
                accumulated_pseudo_ds = None
                
                # Inner Loop: Iterative Chunk Processing
                chunk_results = []
                for i in range(n_chunks):
                    iter_num = i + 1
                    print(f"   [Chunks {n_chunks} | Thresh {thresh_str}] Iteration {iter_num}/{n_chunks}")
                    
                    chunk_idx = chunks[i]
                    ds_chunk = RMLDataset(X_all, Y_all, Z_all, chunk_idx, mode=data_mode)
                    dl_chunk = DataLoader(ds_chunk, batch_size=BATCH_SIZE, shuffle=False)
                    
                    conf_idx, pseudo_labels, chunk_size, oracle_acc, snr_oracle = get_pseudo_labels_detailed(current_model, dl_chunk, threshold)
                    
                    stats = {
                        "config_chunks": n_chunks,
                        "threshold": threshold,
                        "iteration": iter_num,
                        "total_pool_size": chunk_size,
                        "pseudo_labeled_count": len(conf_idx),
                        "pseudo_labeled_percentage": (len(conf_idx)/chunk_size)*100 if chunk_size>0 else 0,
                        "oracle_accuracy_overall": oracle_acc,
                        "oracle_accuracy_per_snr": snr_oracle
                    }
                    print(f"    Stats: Count={len(conf_idx)} Oracle={oracle_acc:.4f}")
                    
                    if len(conf_idx) > 0:
                        real_global_indices = [chunk_idx[loc] for loc in conf_idx]
                        X_new = X_all[real_global_indices]
                        ds_new = PseudoLabelDataset(X_new, pseudo_labels, mode=data_mode)
                        
                        if accumulated_pseudo_ds is None:
                            accumulated_pseudo_ds = ds_new
                        else:
                            accumulated_pseudo_ds = ConcatDataset([accumulated_pseudo_ds, ds_new])
                    
                    if accumulated_pseudo_ds is not None:
                        ds_combined = ConcatDataset([ds_base_train, accumulated_pseudo_ds])
                        dl_combined = DataLoader(ds_combined, batch_size=BATCH_SIZE, shuffle=True)
                        
                        # Save Checkpoint in dedicated folder
                        retrain_path = os.path.join(pth_dir, f'ssl_c{n_chunks}_t{thresh_str}_i{iter_num}.pth')
                        hist_ssl = train_model(current_model, dl_combined, dl_base_val, 
                                               RETRAIN_EPOCHS, retrain_path, patience=5)
                        
                        current_model.load_state_dict(torch.load(retrain_path))
                        metrics = evaluate_model(current_model, dl_test_final, f"{model_name}_SSL_C{n_chunks}_T{thresh_str}_I{iter_num}")
                        
                        chunk_results.append({
                            "stats": stats,
                            "metrics": metrics,
                            "history": hist_ssl
                        })
                    else:
                        chunk_results.append({"stats": stats, "metrics": None, "history": None})

                # Save Individual JSON file for this Threshold
                threshold_json_path = os.path.join(chunk_dir, f'threshold_{thresh_str}.json')
                experiment_data = {
                    "chunks_config": n_chunks,
                    "threshold": threshold,
                    "iterations_data": chunk_results
                }
                save_json(experiment_data, threshold_json_path)
                print(f"    Saved threshold data -> {threshold_json_path}")

        print(f"Finished {model_name}.")

if __name__ == '__main__':
    run_ssl_pipeline()
