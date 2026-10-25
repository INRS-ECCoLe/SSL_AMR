"""Model definitions for "Physics-Informed Representations for Semi-Supervised
Automatic Modulation Classification".

Every model takes a raw I/Q batch of shape (B, 1024, 2) and returns logits of
shape (B, 24). Physics-informed (PI) variants compute amplitude and phase
internally with get_physics_features(), so all models share one input format.

Paired architectures (traditional vs. physics-informed):
    CNN          / PI_CNN           widths {256,128,64,64} / {192,96,48,48}
    Bi_LSTM      / PI_Bi_LSTM       2-layer BiLSTM, hidden 128, input dim 2 / 4
    PETCGDNN     / PI_Bi_PETCGDNN   phase correction + conv + BiGRU
Recent methods (re-implemented from their published descriptions):
    BiStream_Attn_CNNLSTM  Kumar et al., IEEE Commun. Lett., 2024
    CANRN                  Bai et al., ICT Express, 2023
    CSPMNet                Zhang et al., arXiv:2605.25099, 2026
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_physics_features(x):
    """Calculates Amplitude and Phase from I/Q data and appends them to create a 4D feature set."""
    i = x[:, :, 0]
    q = x[:, :, 1]
    amp = torch.sqrt(i**2 + q**2 + 1e-8)
    phase = torch.atan2(q, i)
    return torch.stack((i, q, amp, phase), dim=-1)


# =============================================================================
# Traditional and physics-informed pairs
# =============================================================================
class CNN(nn.Module):
    """Traditional CNN (Baseline ~1.77M Parameters)"""
    def __init__(self, num_classes=24, dr=0.5):
        super(CNN, self).__init__()
        self.pad = nn.ZeroPad2d((3, 4, 0, 1))
        
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(2, 8), padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop = nn.Dropout(dr)
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(2, 8), padding=0)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(2, 8), padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(2, 8), padding=0)
        
        self.fc1 = nn.Linear(64 * 2 * 64, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1) # [B, 1, 2, 128]
        
        x = self.pad(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.pad(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.pad(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.pad(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PI_CNN(nn.Module):
    """Physics-Informed CNN (Capacity matched to ~1.11x of traditional CNN)"""
    def __init__(self, num_classes=24, dr=0.5):
        super(PI_CNN, self).__init__()
        self.pad = nn.ZeroPad2d((3, 4, 0, 1))
        
        self.conv1 = nn.Conv2d(1, 192, kernel_size=(2, 8), padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop = nn.Dropout(dr)
        
        self.conv2 = nn.Conv2d(192, 96, kernel_size=(2, 8), padding=0)
        self.conv3 = nn.Conv2d(96, 48, kernel_size=(2, 8), padding=0)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=(2, 8), padding=0)
        
        self.fc1 = nn.Linear(48 * 4 * 64, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_pi = get_physics_features(x)
        x = x_pi.permute(0, 2, 1).unsqueeze(1) # [B, 1, 4, 128]
        
        x = self.pad(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.pad(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.pad(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.pad(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. Bi-LSTM Variations ---
class Bi_LSTM(nn.Module):
    """Traditional Bi-LSTM (Input: 2 channels)"""
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, num_classes=24, dr=0.5):
        super(Bi_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dr if num_layers > 1 else 0.0, bidirectional=True)
        self.dropout = nn.Dropout(dr)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x is [B, 128, 2]
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class PI_Bi_LSTM(nn.Module):
    """Physics-Informed Bi-LSTM (Input: 4 channels)"""
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, num_classes=24, dr=0.5):
        super(PI_Bi_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dr if num_layers > 1 else 0.0, bidirectional=True)
        self.dropout = nn.Dropout(dr)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        x_pi = get_physics_features(x) # x_pi is [B, 128, 4]
        lstm_out, (h_n, c_n) = self.lstm(x_pi)
        
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# --- 3. PETCGDNN Variations ---
class PETCGDNN(nn.Module):
    """Traditional PETCGDNN (Uses Phase Extraction but keeps 2 channel input depth)"""
    def __init__(self, num_classes=24, dr=0.5):
        super(PETCGDNN, self).__init__()
        self.phase_fc = nn.Linear(1024 * 2, 1)
        
        # Kernel height is 2 to process only rotated I/Q
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(8, 2), padding='valid')
        self.conv1_2 = nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(5, 1), padding='valid')
        
        self.gru = nn.GRU(input_size=25, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dr)
        self.fc_out = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x_iq = x.unsqueeze(1)
        x_i = x[:, :, 0]
        x_q = x[:, :, 1]
        
        x_flat = torch.flatten(x_iq, 1)
        phase_offset = self.phase_fc(x_flat)
        cos_val, sin_val = torch.cos(phase_offset), torch.sin(phase_offset)
        y1, y2 = (x_i * cos_val) + (x_q * sin_val), (x_q * cos_val) - (x_i * sin_val)
        
        # Only use Rotated I and Rotated Q (No Amp/Phase added)
        x_corrected = torch.cat([y1.unsqueeze(-1), y2.unsqueeze(-1)], dim=2)
        x3 = x_corrected.unsqueeze(1)
        
        x3 = F.relu(self.conv1_1(x3))
        x3 = F.relu(self.conv1_2(x3))
        x4 = x3.squeeze(-1).permute(0, 2, 1)
        
        gru_out, h_n = self.gru(x4)
        
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(out)
        return self.fc_out(out)

class PI_Bi_PETCGDNN(nn.Module):
    """Physics-Informed PETCGDNN (Uses Phase Extraction + Amp/Phase to form 4 channel input)"""
    def __init__(self, num_classes=24, dr=0.5):
        super(PI_Bi_PETCGDNN, self).__init__()
        self.phase_fc = nn.Linear(1024 * 2, 1)
        
        # Kernel height is 4 to process rotated I/Q + Amp + Phase
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(8, 4), padding='valid')
        self.conv1_2 = nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(5, 1), padding='valid')
        
        self.gru = nn.GRU(input_size=25, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dr)
        self.fc_out = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x_iq = x.unsqueeze(1)
        x_i = x[:, :, 0]
        x_q = x[:, :, 1]
        
        x_flat = torch.flatten(x_iq, 1)
        phase_offset = self.phase_fc(x_flat)
        cos_val, sin_val = torch.cos(phase_offset), torch.sin(phase_offset)
        y1, y2 = (x_i * cos_val) + (x_q * sin_val), (x_q * cos_val) - (x_i * sin_val)
        
        amp = torch.sqrt(y1**2 + y2**2 + 1e-8)
        phase = torch.atan2(y2, y1)
        
        x_corrected = torch.cat([y1.unsqueeze(-1), y2.unsqueeze(-1), amp.unsqueeze(-1), phase.unsqueeze(-1)], dim=2)
        x3 = x_corrected.unsqueeze(1)
        
        x3 = F.relu(self.conv1_1(x3))
        x3 = F.relu(self.conv1_2(x3))
        x4 = x3.squeeze(-1).permute(0, 2, 1)
        
        gru_out, h_n = self.gru(x4)
        
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(out)
        return self.fc_out(out)

# =============================================================================
# Recent methods
# =============================================================================
class CSPMNet(nn.Module):
    def __init__(self, num_classes=24, dr=0.5):
        super(CSPMNet, self).__init__()
        # Learnable complex phase motion parameter
        self.phase_bias = nn.Parameter(torch.zeros(1))
        
        # Subband feature extractors
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(128 * 256, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(dr)

    def forward(self, x):
        # x is [B, 1024, 2]. Extract I and Q
        i = x[:, :, 0]
        q = x[:, :, 1]
        
        # Apply learnable complex phase motion (Rotation matrix equivalent)
        cos_p = torch.cos(self.phase_bias)
        sin_p = torch.sin(self.phase_bias)
        i_rot = i * cos_p - q * sin_p
        q_rot = i * sin_p + q * cos_p
        
        # Stack back to [B, 2, 1024] for Conv1D
        x_rot = torch.stack([i_rot, q_rot], dim=1) 
        
        x_conv = self.pool(F.relu(self.conv1(x_rot)))
        x_conv = self.pool(F.relu(self.conv2(x_conv)))
        
        x_flat = x_conv.flatten(start_dim=1)
        x_dense = self.drop(F.relu(self.fc1(x_flat)))
        return self.fc2(x_dense)


# [B] Bi-Stream and Attention-Based CNN-LSTM Model
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


# [C] CANRN: Convolution Adaptive Noise Reduction Network
class SoftThresholding(nn.Module):
    def __init__(self, channels):
        super(SoftThresholding, self).__init__()
        # Learnable threshold per channel
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1))
        
    def forward(self, x):
        tau = torch.abs(self.threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - tau)

class CANRN(nn.Module):
    def __init__(self, num_classes=24, dr=0.5):
        super(CANRN, self).__init__()
        # Denoising block
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.denoise = SoftThresholding(64)
        
        # Adaptive feature extraction
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 256, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(dr)

    def forward(self, x):
        # x is [B, 1024, 2]
        x = x.permute(0, 2, 1) # [B, 2, 1024]
        
        # Adaptive Noise Reduction Module
        x = self.conv1(x)
        x = self.denoise(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Processing
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        x = x.flatten(start_dim=1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# name -> (class, group)
MODEL_REGISTRY = {
    "Bi_LSTM":               (Bi_LSTM,               "traditional"),
    "PI_Bi_LSTM":            (PI_Bi_LSTM,            "physics-informed"),
    "PETCGDNN":              (PETCGDNN,              "traditional"),
    "PI_Bi_PETCGDNN":        (PI_Bi_PETCGDNN,        "physics-informed"),
    "CNN":                   (CNN,                   "traditional"),
    "PI_CNN":                (PI_CNN,                "physics-informed"),
    "BiStream_Attn_CNNLSTM": (BiStream_Attn_CNNLSTM, "recent"),
    "CANRN":                 (CANRN,                 "recent"),
    "CSPMNet":               (CSPMNet,               "recent"),
}


def build_model(name, num_classes=24, dr=0.5):
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name][0](num_classes=num_classes, dr=dr)
