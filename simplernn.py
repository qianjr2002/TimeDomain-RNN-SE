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
        
        self.spatial_filters = nn.Parameter(torch.empty(hidden_dim, n_channels))
        nn.init.kaiming_uniform_(self.spatial_filters)
        
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

if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    print("Table 3:")
    print("L=4ms")
    model_L4_C2_H300_a  = SimpleRNNModel(n_channels=2, hidden_dim=300, iW=64, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C2_H300_a, (2, 16000),print_per_layer_stat=False)
    print(f"model_L4_C2_H300_a flops {flops} params {params}")
    # model_L4_C2_H300_a flops 2.23 GMac params 2.21 M

    model_L4_C4_H300_a  = SimpleRNNModel(n_channels=4, hidden_dim=300, iW=64, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C4_H300_a, (4, 16000),print_per_layer_stat=False)
    print(f"model_L4_C4_H300_a flops {flops} params {params}")
    # model_L4_C4_H300_a flops 2.27 GMac params 2.21 M

    model_L4_C8_H300_a  = SimpleRNNModel(n_channels=8, hidden_dim=300, iW=64, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C8_H300_a, (8, 16000),print_per_layer_stat=False)
    print(f"model_L4_C8_H300_a flops {flops} params {params}")
    # model_L4_C8_H300_a flops 2.35 GMac params 2.21 M

    model_L4_C2_H300_b  = SimpleRNNModel(n_channels=2, hidden_dim=300, iW=256, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C2_H300_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_L4_C2_H300_b flops {flops} params {params}")
    # model_L4_C2_H300_b flops 2.35 GMac params 2.27 M

    model_L4_C4_H300_b  = SimpleRNNModel(n_channels=4, hidden_dim=300, iW=256, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C4_H300_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_L4_C4_H300_b flops {flops} params {params}")
    # model_L4_C4_H300_b flops 2.5 GMac params 2.27 M

    model_L4_C8_H300_b  = SimpleRNNModel(n_channels=8, hidden_dim=300, iW=256, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C8_H300_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_L4_C8_H300_b flops {flops} params {params}")
    # model_L4_C8_H300_b flops 2.81 GMac params 2.27 M

    model_L4_C2_H1024_b = SimpleRNNModel(n_channels=2, hidden_dim=1024, iW=256, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C2_H1024_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_L4_C2_H1024_b flops {flops} params {params}")
    # model_L4_C2_H1024_b flops 25.74 GMac params 25.53 M

    model_L4_C4_H1024_b = SimpleRNNModel(n_channels=4, hidden_dim=1024, iW=256, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C4_H1024_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_L4_C4_H1024_b flops {flops} params {params}")
    # model_L4_C4_H1024_b flops 26.28 GMac params 25.53 M

    model_L4_C8_H1024_b = SimpleRNNModel(n_channels=8, hidden_dim=1024, iW=256, oW=64, S=16, B=3)
    flops, params = get_model_complexity_info(model_L4_C8_H1024_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_L4_C8_H1024_b flops {flops} params {params}")
    # model_L4_C8_H1024_b flops 27.34 GMac params 25.54 M

    print("L=2ms")
    model_L2_C2_H300_a  = SimpleRNNModel(n_channels=2, hidden_dim=300, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C2_H300_a, (2, 16000),print_per_layer_stat=False)
    print(f"model_L2_C2_H300_a flops {flops} params {params}")
    # model_L2_C2_H300_a flops 2.21 GMac params 2.19 M

    model_L2_C4_H300_a  = SimpleRNNModel(n_channels=4, hidden_dim=300, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C4_H300_a, (4, 16000),print_per_layer_stat=False)
    print(f"model_L2_C4_H300_a flops {flops} params {params}")
    # model_L2_C4_H300_a flops 2.23 GMac params 2.19 M

    model_L2_C8_H300_a  = SimpleRNNModel(n_channels=8, hidden_dim=300, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C8_H300_a, (8, 16000),print_per_layer_stat=False)
    print(f"model_L2_C8_H300_a flops {flops} params {params}")
    # model_L2_C8_H300_a flops 2.27 GMac params 2.19 M

    model_L2_C2_H300_b  = SimpleRNNModel(n_channels=2, hidden_dim=300, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C2_H300_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_L2_C2_H300_b flops {flops} params {params}")
    # model_L2_C2_H300_b flops 2.34 GMac params 2.26 M

    model_L2_C4_H300_b  = SimpleRNNModel(n_channels=4, hidden_dim=300, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C4_H300_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_L2_C4_H300_b flops {flops} params {params}")
    # model_L2_C4_H300_b flops 2.5 GMac params 2.26 M

    model_L2_C8_H300_b  = SimpleRNNModel(n_channels=8, hidden_dim=300, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C8_H300_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_L2_C8_H300_b flops {flops} params {params}")
    # model_L2_C8_H300_b flops 2.81 GMac params 2.26 M


    model_L2_C2_H1024_b = SimpleRNNModel(n_channels=2, hidden_dim=1024, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C2_H1024_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_L2_C2_H1024_b flops {flops} params {params}")
    # model_L2_C2_H1024_b flops 25.76 GMac params 25.5 M

    model_L2_C4_H1024_b = SimpleRNNModel(n_channels=4, hidden_dim=1024, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C4_H1024_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_L2_C4_H1024_b flops {flops} params {params}")
    # model_L2_C4_H1024_b flops 26.3 GMac params 25.5 M

    model_L2_C8_H1024_b = SimpleRNNModel(n_channels=8, hidden_dim=1024, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_L2_C8_H1024_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_L2_C8_H1024_b flops {flops} params {params}")
    # model_L2_C8_H1024_b flops 27.36 GMac params 25.5 M

    print("L=1ms")
    model_L1_C2_H300_a  = SimpleRNNModel(n_channels=2, hidden_dim=300, iW=16, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C2_H300_a, (2, 16000),print_per_layer_stat=False)
    print(f"model_L1_C2_H300_a flops {flops} params {params}")
    # model_L1_C2_H300_a flops 2.19 GMac params 2.18 M

    model_L1_C4_H300_a  = SimpleRNNModel(n_channels=4, hidden_dim=300, iW=16, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C4_H300_a, (4, 16000),print_per_layer_stat=False)
    print(f"model_L1_C4_H300_a flops {flops} params {params}")
    # model_L1_C4_H300_a flops 2.21 GMac params 2.18 M

    model_L1_C8_H300_a  = SimpleRNNModel(n_channels=8, hidden_dim=300, iW=16, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C8_H300_a, (8, 16000),print_per_layer_stat=False)
    print(f"model_L1_C8_H300_a flops {flops} params {params}")
    # model_L1_C8_H300_a flops 2.23 GMac params 2.18 M

    model_L1_C2_H300_b  = SimpleRNNModel(n_channels=2, hidden_dim=300, iW=256, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C2_H300_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_L1_C2_H300_b flops {flops} params {params}")
    # model_L1_C2_H300_b flops 2.34 GMac params 2.25 M

    model_L1_C4_H300_b  = SimpleRNNModel(n_channels=4, hidden_dim=300, iW=256, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C4_H300_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_L1_C4_H300_b flops {flops} params {params}")
    # model_L1_C4_H300_b flops 2.49 GMac params 2.25 M

    model_L1_C8_H300_b  = SimpleRNNModel(n_channels=8, hidden_dim=300, iW=256, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C8_H300_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_L1_C8_H300_b flops {flops} params {params}")
    # model_L1_C8_H300_b flops 2.81 GMac params 2.25 M

    model_L1_C2_H1024_b = SimpleRNNModel(n_channels=2, hidden_dim=1024, iW=256, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C2_H1024_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_L1_C2_H1024_b flops {flops} params {params}")
    # model_L1_C2_H1024_b flops 25.77 GMac params 25.48 M

    model_L1_C4_H1024_b = SimpleRNNModel(n_channels=4, hidden_dim=1024, iW=256, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C4_H1024_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_L1_C4_H1024_b flops {flops} params {params}")
    # model_L1_C4_H1024_b flops 26.31 GMac params 25.48 M

    model_L1_C8_H1024_b = SimpleRNNModel(n_channels=8, hidden_dim=1024, iW=256, oW=16, S=16, B=3)
    flops, params = get_model_complexity_info(model_L1_C8_H1024_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_L1_C8_H1024_b flops {flops} params {params}")
    # model_L1_C8_H1024_b flops 27.37 GMac params 25.49 M

    # 模式,iW (输入窗),oW (输出窗)
    # Approach (a),L×16,L×16
    # Approach (b),256 (固定 16ms),L×16
    model_T1_L1_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=16, oW=16, S=16, B=3)
    model_T1_L1_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=256, oW=16, S=16, B=3)
    model_T1_L1_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=16, oW=16, S=16, B=3)
    model_T1_L1_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=256, oW=16, S=16, B=3)
    model_T1_L1_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=16, oW=16, S=16, B=3)
    model_T1_L1_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=256, oW=16, S=16, B=3)

    model_T1_L2_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=32, oW=32, S=16, B=3)
    model_T1_L2_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=256, oW=32, S=16, B=3)
    model_T1_L2_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=32, oW=32, S=16, B=3)
    model_T1_L2_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=256, oW=32, S=16, B=3)
    model_T1_L2_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=32, oW=32, S=16, B=3)
    model_T1_L2_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=256, oW=32, S=16, B=3)

    model_T1_L4_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=64, oW=64, S=16, B=3)
    model_T1_L4_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=256, oW=64, S=16, B=3)
    model_T1_L4_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=64, oW=64, S=16, B=3)
    model_T1_L4_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=256, oW=64, S=16, B=3)
    model_T1_L4_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=64, oW=64, S=16, B=3)
    model_T1_L4_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=256, oW=64, S=16, B=3)

    model_T1_L8_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=128, oW=128, S=16, B=3)
    model_T1_L8_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=256, oW=128, S=16, B=3)
    model_T1_L8_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=128, oW=128, S=16, B=3)
    model_T1_L8_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=256, oW=128, S=16, B=3)
    model_T1_L8_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=128, oW=128, S=16, B=3)
    model_T1_L8_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=256, oW=128, S=16, B=3)

    model_T1_L16_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=256, oW=256, S=16, B=3)
    model_T1_L16_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=256, oW=256, S=16, B=3)
    model_T1_L16_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=256, oW=256, S=16, B=3)
    model_T1_L16_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=256, oW=256, S=16, B=3)
    model_T1_L16_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=256, oW=256, S=16, B=3)
    model_T1_L16_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=256, oW=256, S=16, B=3)



    print("Table 2:")
    # Approach      iW (输入窗),oW (输出窗)
    # (a),32 (fixed 2ms)      ,32
    # (b),256 (fixed 16ms)     ,32
    model_T2_H64_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=64, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H64_C2_a, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H64_C2_a flops {flops} params {params}")
    # model_T2_H64_C2_a flops 108.53 MMac params 104.67 k

    model_T2_H64_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=64, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H64_C2_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H64_C2_b flops {flops} params {params}")
    # model_T2_H64_C2_b flops 137.17 MMac params 119.01 k

    model_T2_H64_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=64, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H64_C4_a, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H64_C4_a flops {flops} params {params}")
    # model_T2_H64_C4_a flops 113.13 MMac params 104.8 k
    
    model_T2_H64_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=64, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H64_C4_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H64_C4_b flops {flops} params {params}")
    # model_T2_H64_C4_b flops 170.42 MMac params 119.14 k
    
    model_T2_H64_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=64, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H64_C8_a, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H64_C8_a flops {flops} params {params}")
    # model_T2_H64_C8_a flops 122.34 MMac params 105.06 k
    
    model_T2_H64_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=64, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H64_C8_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H64_C8_b flops {flops} params {params}")
    # model_T2_H64_C8_b flops 236.91 MMac params 119.39 k

    model_T2_H128_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=128, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H128_C2_a, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H128_C2_a flops {flops} params {params}")
    # model_T2_H128_C2_a flops 413.44 MMac params 405.92 k
    
    model_T2_H128_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=128, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H128_C2_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H128_C2_b flops {flops} params {params}")
    # model_T2_H128_C2_b flops 470.73 MMac params 434.59 k
    
    model_T2_H128_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=128, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H128_C4_a, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H128_C4_a flops {flops} params {params}")
    # model_T2_H128_C4_a flops 422.65 MMac params 406.18 k

    model_T2_H128_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=128, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H128_C4_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H128_C4_b flops {flops} params {params}")
    # model_T2_H128_C4_b flops 537.22 MMac params 434.85 k

    model_T2_H128_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=128, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H128_C8_a, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H128_C8_a flops {flops} params {params}")
    # model_T2_H128_C8_a flops 441.06 MMac params 406.69 k

    model_T2_H128_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=128, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H128_C8_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H128_C8_b flops {flops} params {params}")
    # model_T2_H128_C8_b flops 670.21 MMac params 435.36 k

    model_T2_H256_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H256_C2_a, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H256_C2_a flops {flops} params {params}")
    # model_T2_H256_C2_a flops 1.61 GMac params 1.6 M

    model_T2_H256_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=256, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H256_C2_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H256_C2_b flops {flops} params {params}")
    # model_T2_H256_C2_b flops 1.73 GMac params 1.66 M

    model_T2_H256_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H256_C4_a, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H256_C4_a flops {flops} params {params}")
    # model_T2_H256_C4_a flops 1.63 GMac params 1.6 M

    model_T2_H256_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=256, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H256_C4_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H256_C4_b flops {flops} params {params}")
    # model_T2_H256_C4_b flops 1.86 GMac params 1.66 M

    model_T2_H256_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H256_C8_a, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H256_C8_a flops {flops} params {params}")
    # model_T2_H256_C8_a flops 1.67 GMac params 1.6 M

    model_T2_H256_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=256, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H256_C8_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H256_C8_b flops {flops} params {params}")
    # model_T2_H256_C8_b flops 2.13 GMac params 1.66 M

    model_T2_H512_C2_a = SimpleRNNModel(n_channels=2, hidden_dim=512, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H512_C2_a, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H512_C2_a flops {flops} params {params}")
    # model_T2_H512_C2_a flops 6.37 GMac params 6.34 M

    model_T2_H512_C2_b = SimpleRNNModel(n_channels=2, hidden_dim=512, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H512_C2_b, (2, 16000),print_per_layer_stat=False)
    print(f"model_T2_H512_C2_b flops {flops} params {params}")
    # model_T2_H512_C2_b flops 6.6 GMac params 6.46 M

    model_T2_H512_C4_a = SimpleRNNModel(n_channels=4, hidden_dim=512, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H512_C4_a, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H512_C4_a flops {flops} params {params}")
    # model_T2_H512_C4_a flops 6.4 GMac params 6.34 M

    model_T2_H512_C4_b = SimpleRNNModel(n_channels=4, hidden_dim=512, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H512_C4_b, (4, 16000),print_per_layer_stat=False)
    print(f"model_T2_H512_C4_b flops {flops} params {params}")
    # model_T2_H512_C4_b flops 6.86 GMac params 6.46 M

    model_T2_H512_C8_a = SimpleRNNModel(n_channels=8, hidden_dim=512, iW=32, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H512_C8_a, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H512_C8_a flops {flops} params {params}")
    # model_T2_H512_C8_a flops 6.48 GMac params 6.35 M

    model_T2_H512_C8_b = SimpleRNNModel(n_channels=8, hidden_dim=512, iW=256, oW=32, S=16, B=3)
    flops, params = get_model_complexity_info(model_T2_H512_C8_b, (8, 16000),print_per_layer_stat=False)
    print(f"model_T2_H512_C8_b flops {flops} params {params}")
    # model_T2_H512_C8_b flops 7.39 GMac params 6.46 M

