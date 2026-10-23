"""
STL-Net + physics-guided frequency attention.

Backbone: STL-Net, i.e. multi-scale seasonal-trend decomposition with trend,
seasonal and residual branches.
Extra: frequency attention guided by bearing fault frequencies.
RPM enters the physics branch only, there is no learned speed encoder.

Flow:
  input (B, 5, 2048)
    ├─ Conv1d projection 5 → 32
    ├─ multi-scale decomposition, kernels 19 / 31 / 117
    │    ├─ trend branch (large-kernel CNN)                 → 128
    │    ├─ seasonal branch (small-kernel CNN + attention)  → 128
    │    └─ residual branch (x_proj - trend at k=117)       → 64
    ├─ physics frequency branch: rFFT per sensor (4 sensors)
    │    rpm → bpfo / bpfi / bsf / ftf plus 3 harmonics
    │    → gaussian mask ⊙ learned mask, gated              → 128
    └─ concat 320 + 128 = 448 → classifier → num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # (B, L, C) -> (B, L, C)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return trend


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class MultiScaleDecomp(nn.Module):
    """
    Three decomposition scales for the 16 kHz gearbox signal:
      kernel=19  ~1.19 ms  shortest fault-period reference
      kernel=31  ~1.94 ms  representative outer-race fault period
      kernel=117 ~7.31 ms  shortest shaft-rotation period
    """

    def __init__(self, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [19, 31, 117]  # kinematic reference scales at fs=16 kHz
        self.kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.decomposers = nn.ModuleList([SeriesDecomp(k) for k in self.kernel_sizes])

    def forward(self, x):  # (B, L, C) -> seasonal list, trend list
        seasonal_list, trend_list = [], []
        for decomposer in self.decomposers:
            seasonal, trend = decomposer(x)
            seasonal_list.append(seasonal)
            trend_list.append(trend)
        return seasonal_list, trend_list


class TrendBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, trend_list):  # list of (B, L, C) -> (B, out_channels)
        trends = torch.cat(trend_list, dim=-1).permute(0, 2, 1)
        return self.pool(self.conv_layers(trends)).squeeze(-1)


class SeasonalBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=128,
                 n_heads=4, seq_len=2048):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.downsample = nn.Conv1d(hidden_channels, hidden_channels,
                                    kernel_size=4, stride=4)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels, num_heads=n_heads,
            dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, out_channels),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, seasonal_list):  # list of (B, L, C) -> (B, out_channels)
        seasonals = torch.cat(seasonal_list, dim=-1).permute(0, 2, 1)
        x = self.conv_layers(seasonals)
        x = self.downsample(x).permute(0, 2, 1)   # 2048 -> 512 for the attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_out = self.ffn(self.norm2(x))
        return self.pool(x_out.permute(0, 2, 1)).squeeze(-1)


class ResidualBranch(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):  # (B, C, L) -> (B, out_channels)
        return self.conv(x).squeeze(-1)


# fault characteristic frequencies from bearing geometry and rpm
class BearingPhysicsCalculator:
    def __init__(self, bearing_params=None):
        if bearing_params is None:
            bearing_params = {
                'n_balls': 11,
                'ball_diameter': 7.938,
                'pitch_diameter': 48.5,
                'contact_angle': 0.0
            }
        self.n_balls = bearing_params['n_balls']
        self.ball_diameter = bearing_params['ball_diameter']
        self.pitch_diameter = bearing_params['pitch_diameter']
        self.contact_angle = bearing_params['contact_angle']

    def calculate_fault_frequencies(self, rpm):
        if isinstance(rpm, torch.Tensor):
            rpm_np = rpm.cpu().detach().numpy()
            is_tensor, device = True, rpm.device
        else:
            rpm_np = np.asarray(rpm)
            is_tensor, device = False, None

        shaft_freq = rpm_np / 60.0
        ratio = self.ball_diameter / self.pitch_diameter
        cos_a = np.cos(self.contact_angle)

        freqs = {
            'bpfo': (self.n_balls / 2) * shaft_freq * (1 - ratio * cos_a),
            'bpfi': (self.n_balls / 2) * shaft_freq * (1 + ratio * cos_a),
            'bsf': (self.pitch_diameter / (2 * self.ball_diameter)) * shaft_freq * (1 - (ratio * cos_a) ** 2),
            'ftf': (shaft_freq / 2) * (1 - ratio * cos_a),
        }

        if is_tensor:
            return {k: torch.from_numpy(v).float().to(device) for k, v in freqs.items()}
        return freqs


class PhysicsGuidedFrequencyAttention(nn.Module):
    """
    Blend a physics-derived mask with a learned mask through a match gate.
    match_gate ~ 0 for healthy windows (data-driven), ~ 1 for faulty ones.
    """

    def __init__(self, freq_bins=1025, fs=16000, window_size=2048):
        super().__init__()
        self.freq_bins = freq_bins
        self.fs = fs
        self.freq_resolution = fs / window_size
        self.physics_calc = BearingPhysicsCalculator()

        # learnable parameters
        self.gaussian_sigma = nn.Parameter(torch.ones(4) * 10.0)
        self.fault_type_weights = nn.Parameter(torch.ones(4))
        self.harmonic_decay = nn.Parameter(torch.tensor([1.0, 0.8, 0.6]))

        # data-driven attention
        self.learned_attention = nn.Sequential(
            nn.Linear(freq_bins, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, freq_bins), nn.Sigmoid()
        )
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # temperature controls how sharp match_score is
        self.match_temperature = nn.Parameter(torch.tensor(5.0))
        # below this value the window is treated as healthy and the physics mask is skipped
        self.match_threshold = nn.Parameter(torch.tensor(0.3))

    def _create_physics_mask(self, magnitude, rpm):
        B, device = magnitude.shape[0], magnitude.device
        fault_freqs = self.physics_calc.calculate_fault_frequencies(rpm)

        freq_axis = torch.arange(self.freq_bins, device=device).float() * self.freq_resolution
        freq_axis = freq_axis.unsqueeze(0).expand(B, -1)

        fault_types = ['bpfo', 'bpfi', 'bsf', 'ftf']
        base_freqs = torch.stack([
            fault_freqs[ft] if fault_freqs[ft].ndim > 0 else fault_freqs[ft].expand(B)
            for ft in fault_types
        ], dim=1).to(device)

        harmonics = torch.arange(1, 4, device=device).float()
        all_freqs = (base_freqs.unsqueeze(2) * harmonics.view(1, 1, 3)).reshape(B, -1)

        freq_diff = freq_axis.unsqueeze(1) - all_freqs.unsqueeze(2)
        sigmas = (torch.abs(self.gaussian_sigma).repeat_interleave(3).unsqueeze(0).unsqueeze(2) + 1e-6)
        gaussian = torch.exp(-(freq_diff ** 2) / (2 * sigmas ** 2))

        weights = (self.harmonic_decay.repeat(4) * torch.abs(self.fault_type_weights).repeat_interleave(3))
        gaussian = gaussian * weights.unsqueeze(0).unsqueeze(2)

        nyquist = self.fs / 2
        valid = ((all_freqs > 0) & (all_freqs < nyquist)).unsqueeze(2).float()
        mask = (gaussian * valid).sum(dim=1)
        mask = mask / (mask.max(dim=1, keepdim=True)[0] + 1e-8)
        return mask

    def forward(self, magnitude, rpm):
        physics_mask = self._create_physics_mask(magnitude, rpm)
        learned_mask = self.learned_attention(magnitude)

        # how well the physics mask lines up with the observed spectrum
        match_score = (magnitude * physics_mask).sum(dim=-1, keepdim=True) / (
            magnitude.sum(dim=-1, keepdim=True) + 1e-8)

        # temperature scaling + threshold, so the gate is ~0 for healthy windows
        # (pure data-driven) and ~1 for faulty ones (physics mask dominates)
        match_gate = torch.sigmoid(
            self.match_temperature * (match_score - torch.sigmoid(self.match_threshold))
        )

        alpha = torch.sigmoid(self.fusion_weight) * match_gate
        combined = alpha * physics_mask + (1 - alpha) * learned_mask
        return magnitude * combined


class PhysicsGuidedFrequencyEncoder(nn.Module):
    def __init__(self, window_size=2048, fs=16000, num_freq_bins=256):
        super().__init__()
        self.fft_bins = window_size // 2 + 1
        self.physics_attention = PhysicsGuidedFrequencyAttention(
            freq_bins=self.fft_bins, fs=fs, window_size=window_size
        )
        self.freq_compressor = nn.Linear(self.fft_bins, num_freq_bins)

    def forward(self, x, rpm):  # x: (B, window_size), rpm: (B,) -> (B, num_freq_bins)
        fft_result = torch.fft.rfft(x, dim=-1)
        magnitude = torch.sqrt(fft_result.real ** 2 + fft_result.imag ** 2)
        weighted = self.physics_attention(magnitude, rpm)
        return self.freq_compressor(torch.log1p(weighted))


# not wired into the model any more, kept for reference
class SpeedEncoder(nn.Module):
    def __init__(self, window_size=2048, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 64), nn.ReLU(), nn.Linear(64, hidden_dim)
        )
        self.register_buffer('rpm_mean', torch.tensor(0.0))
        self.register_buffer('rpm_std', torch.tensor(1.0))

    def set_global_rpm_stats(self, mean, std):
        self.rpm_mean = torch.tensor(mean, dtype=torch.float32)
        self.rpm_std = torch.tensor(std, dtype=torch.float32)
        print(f"global rpm stats: mean={mean:.2f}, std={std:.2f}")

    def forward(self, speed_data):
        normalized = (speed_data - self.rpm_mean) / (self.rpm_std + 1e-8)
        return self.encoder(normalized)


class STLPhysicsNet(nn.Module):
    """STL-Net backbone plus the physics frequency branch, 448-d features."""

    def __init__(self, num_sensors=5, window_size=2048, num_classes=4,
                 decomp_kernel_sizes=None,
                 trend_hidden=64, trend_out=128,
                 seasonal_hidden=64, seasonal_out=128,
                 residual_out=64, n_heads=4, dropout=0.2,
                 fs=16000, num_freq_bins=256):
        super().__init__()

        self.num_sensors = num_sensors
        self.window_size = window_size
        self.num_accel_sensors = num_sensors - 1

        if decomp_kernel_sizes is None:
            decomp_kernel_sizes = [19, 31, 117]

        # STL-Net backbone
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_sensors, 32, kernel_size=1),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        proj_ch = 32
        self.num_scales = len(decomp_kernel_sizes)

        self.decomposer = MultiScaleDecomp(kernel_sizes=decomp_kernel_sizes)

        self.trend_branch = TrendBranch(
            in_channels=proj_ch * self.num_scales,
            hidden_channels=trend_hidden, out_channels=trend_out
        )
        self.seasonal_branch = SeasonalBranch(
            in_channels=proj_ch * self.num_scales,
            hidden_channels=seasonal_hidden, out_channels=seasonal_out,
            n_heads=n_heads, seq_len=window_size
        )
        self.residual_branch = ResidualBranch(
            in_channels=proj_ch, out_channels=residual_out
        )

        stl_dim = trend_out + seasonal_out + residual_out  # 320

        # physics frequency branch
        self.freq_extractors = nn.ModuleList([
            PhysicsGuidedFrequencyEncoder(window_size, fs, num_freq_bins)
            for _ in range(self.num_accel_sensors)
        ])
        self.freq_fusion = nn.Sequential(
            nn.Linear(num_freq_bins * self.num_accel_sensors, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128)
        )
        freq_dim = 128

        # classifier
        total_dim = stl_dim + freq_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_rpm_stats(self, mean, std):  # speed branch is gone, kept for callers
        pass

    def forward(self, x):
        # x: (B, 5, 2048), x[:, 0:4] = normalised acceleration, x[:, 4] = raw rpm

        # STL-Net backbone
        x_proj = self.input_proj(x)                          # (B, 32, 2048)
        x_seq = x_proj.permute(0, 2, 1)                      # (B, 2048, 32)

        seasonal_list, trend_list = self.decomposer(x_seq)

        trend_feat = self.trend_branch(trend_list)           # (B, 128)
        seasonal_feat = self.seasonal_branch(seasonal_list)  # (B, 128)

        main_trend = trend_list[-1].permute(0, 2, 1)         # (B, 32, 2048)
        residual = x_proj - main_trend
        residual_feat = self.residual_branch(residual)       # (B, 64)

        stl_features = torch.cat([trend_feat, seasonal_feat, residual_feat], dim=-1)

        # physics frequency branch
        speed_data = x[:, 4, :]
        avg_rpm = speed_data.mean(dim=1)

        freq_features = []
        for i in range(self.num_accel_sensors):
            freq_features.append(self.freq_extractors[i](x[:, i, :], avg_rpm))
        freq_features = self.freq_fusion(torch.cat(freq_features, dim=1))

        # fuse and classify
        combined = torch.cat([stl_features, freq_features], dim=-1)  # 448-d
        return self.classifier(combined)


def create_model(model_type='stl_physics', num_sensors=5, window_size=2048,
                 num_classes=4, **kwargs):
    """Build the model. Only 'stl_physics' is supported."""
    fs = kwargs.get('fs', 16000)
    num_freq_bins = kwargs.get('num_freq_bins', 256)
    dropout = kwargs.get('dropout', 0.2)

    if model_type == 'stl_physics':
        print("building model: STL-Net + physics-guided frequency attention")
        model = STLPhysicsNet(
            num_sensors=num_sensors,
            window_size=window_size,
            num_classes=num_classes,
            fs=fs,
            num_freq_bins=num_freq_bins,
            dropout=dropout
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {num_params:,}")
    return model