import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__(); self.backbone=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,32),nn.ReLU()); self.classifier=nn.Linear(32,3); self.regressor=nn.Linear(32,1)
    def forward(self,x): h=self.backbone(x); return self.classifier(h),self.regressor(h).squeeze(1)

class GRUBaseline(nn.Module):
    def __init__(self,input_dim):
        super().__init__(); self.gru=nn.GRU(input_dim,64,batch_first=True); self.shared=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.15)); self.classifier=nn.Linear(32,3); self.regressor=nn.Linear(32,1)
    def forward(self,x): _,h=self.gru(x); h=self.shared(h[-1]); return self.classifier(h),self.regressor(h).squeeze(1)


class CausalConv1d(nn.Conv1d):
    """A left-padded convolution: output t cannot depend on inputs after t."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=.10):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, 3, dilation)
        self.conv2 = CausalConv1d(channels, channels, 3, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(F.relu(self.conv1(x)))
        h = self.dropout(F.relu(self.conv2(h)))
        return F.relu(x + h)


class TCNCausalBaseline(nn.Module):
    """Small causal, dilated TCN using only the supplied history window."""
    def __init__(self, input_dim, channels=48):
        super().__init__()
        self.input_projection = CausalConv1d(input_dim, channels, 1)
        self.blocks = nn.Sequential(*(TemporalResidualBlock(channels, d) for d in (1, 2, 4)))
        self.shared = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Dropout(.10))
        self.classifier = nn.Linear(32, 3)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        # [batch, time, features] -> read only the representation at current time.
        h = self.blocks(self.input_projection(x.transpose(1, 2)))[:, :, -1]
        h = self.shared(h)
        return self.classifier(h), self.regressor(h).squeeze(1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim, max_length=32):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        table = torch.zeros(max_length, dim)
        table[:, 0::2] = torch.sin(position * scale)
        table[:, 1::2] = torch.cos(position * scale)
        self.register_buffer('table', table.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.table[:, :x.size(1)]


class TransformerCausalBaseline(nn.Module):
    """Lightweight masked Transformer Encoder. The final token is the endpoint."""
    def __init__(self, input_dim, hidden_dim=48, heads=4, layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = SinusoidalPositionEncoding(hidden_dim)
        block = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=96,
                                           dropout=.10, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.shared = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(.10))
        self.classifier = nn.Linear(32, 3)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        length = x.size(1)
        causal_mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(self.position(self.input_projection(x)), mask=causal_mask)[:, -1]
        h = self.shared(h)
        return self.classifier(h), self.regressor(h).squeeze(1)
