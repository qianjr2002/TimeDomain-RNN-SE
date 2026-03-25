import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNNModel(nn.Module):
    def __init__(self, n_channels=8, hidden_dim=256, iW=32, oW=32, S=16, B=3):
        super().__init__()
        self.iW = iW  # input window size (e.g., 16ms = 256 samples)
        self.oW = oW  # output window size (e.g. 2ms = 32 samples)
        self.S = S    # stride (1ms = 16 samples)
        self.H = hidden_dim  # hidden dimension (e.g., 256)
        
        self.input_proj = nn.Sequential(
            nn.Linear(iW, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU()
        )
        
        self.spatial_filters = nn.Parameter(torch.randn(hidden_dim, n_channels))
        
        self.rnn_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'lstm': nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            }) for _ in range(B)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, oW)

    def forward(self, x):
        B, C, N = x.shape
        
        left_pad = self.iW - self.oW
        
        total_pad = (self.S - (N - self.oW) % self.S) % self.S
        x_padded = F.pad(x, (left_pad, total_pad)) 
        
        frames = x_padded.unfold(-1, self.iW, self.S)
        T = frames.size(2)

        x_feat = self.input_proj(frames)
        x_feat = torch.einsum('bcth,hc->bth', x_feat, self.spatial_filters)
        
        for layer in self.rnn_layers:
            x_feat = layer['norm'](x_feat)
            x_feat, _ = layer['lstm'](x_feat)
        
        out_frames = self.output_proj(x_feat)
        
        out_frames = out_frames.transpose(1, 2) # [B, oW, T]
        
        output_size = (1, self.S * (T - 1) + self.oW)
        enhanced = F.fold(
            out_frames,
            output_size=output_size,
            kernel_size=(1, self.oW),
            stride=(1, self.S)
        )
        
        enhanced = enhanced.squeeze(1).squeeze(1) 
        return enhanced[:, :N]
