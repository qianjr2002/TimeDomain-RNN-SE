# TimeDomain-RNN-SE

🚀 Unofficial implementation of [A Simple RNN Model for Lightweight, Low-compute and Low-latency Multichannel Speech Enhancement in the Time Domain](https://www.isca-archive.org/interspeech_2023/pandey23b_interspeech.html) in PyTorch.

## Overview

This project implements a lightweight RNN-based model for real-time multichannel speech enhancement in the time domain. The model is designed for low-compute and low-latency applications, making it suitable for edge devices and real-time processing scenarios.

## Model Architecture

The `SimpleRNNModel` consists of the following components:

- **Input Projection**: Linear layer + LayerNorm + PReLU
- **Spatial Filters**: Learnable spatial attention mechanism for multichannel processing
- **RNN Layers**: Stack of LSTM layers with LayerNorm (default: 3 layers)
- **Output Projection**: Linear layer for enhanced signal reconstruction

### Key Parameters

- `n_channels`: Number of input channels (default: 8)
- `hidden_dim`: Hidden dimension of RNN (default: 256)
- `iW`: Input window size in samples (e.g., 32 samples = 2ms at 16kHz)
- `oW`: Output window size in samples (e.g., 32 samples = 2ms at 16kHz)
- `S`: Stride in samples (default: 16 samples = 1ms at 16kHz)
- `B`: Number of RNN layers (default: 3)

## Usage

### Basic Usage

```python
from simplernn import SimpleRNNModel

# Initialize model
model = SimpleRNNModel(
    n_channels=8,    # Number of input channels
    hidden_dim=256,  # Hidden dimension
    iW=32,           # Input window size (2ms at 16kHz)
    oW=32,           # Output window size (2ms at 16kHz)
    S=16,            # Stride (1ms at 16kHz)
    B=3              # Number of RNN layers
)

# Forward pass
# Input shape: [batch_size, n_channels, num_samples]
# Output shape: [batch_size, num_samples]
enhanced_audio = model(input_audio)
```

### Model Complexity Analysis

The repository includes comprehensive complexity analysis for various model configurations:

```bash
python simplernn.py
```

This will output FLOPs and parameter counts for different model variants.

## Model Configurations

### Table 3: Latency vs. Performance Trade-offs

| Model | Channels | Hidden Dim | Input Window | Output Window | FLOPs | Parameters |
|-------|----------|------------|--------------|---------------|-------|------------|
| L1_C2_H300_a | 2 | 300 | 16 (1ms) | 16 (1ms) | 2.19 GMac | 2.18 M |
| L1_C4_H300_a | 4 | 300 | 16 (1ms) | 16 (1ms) | 2.21 GMac | 2.18 M |
| L1_C8_H300_a | 8 | 300 | 16 (1ms) | 16 (1ms) | 2.23 GMac | 2.18 M |
| L2_C2_H300_a | 2 | 300 | 32 (2ms) | 32 (2ms) | 2.21 GMac | 2.19 M |
| L2_C4_H300_a | 4 | 300 | 32 (2ms) | 32 (2ms) | 2.23 GMac | 2.19 M |
| L2_C8_H300_a | 8 | 300 | 32 (2ms) | 32 (2ms) | 2.27 GMac | 2.19 M |
| L4_C2_H300_a | 2 | 300 | 64 (4ms) | 64 (4ms) | 2.23 GMac | 2.21 M |
| L4_C4_H300_a | 4 | 300 | 64 (4ms) | 64 (4ms) | 2.27 GMac | 2.21 M |
| L4_C8_H300_a | 8 | 300 | 64 (4ms) | 64 (4ms) | 2.35 GMac | 2.21 M |

### Table 2: Model Size Variants

| Model | Channels | Hidden Dim | FLOPs | Parameters |
|-------|----------|------------|-------|------------|
| H64_C2_a | 2 | 64 | 108.53 MMac | 104.67 k |
| H64_C2_b | 2 | 64 | 137.17 MMac | 119.01 k |
| H128_C2_a | 2 | 128 | 413.44 MMac | 405.92 k |
| H128_C2_b | 2 | 128 | 470.73 MMac | 434.59 k |
| H256_C2_a | 2 | 256 | 1.61 GMac | 1.6 M |
| H256_C2_b | 2 | 256 | 1.73 GMac | 1.66 M |
| H512_C2_a | 2 | 512 | 6.37 GMac | 6.34 M |
| H512_C2_b | 2 | 512 | 6.6 GMac | 6.46 M |

## Model Variants

The implementation supports two main approaches:

### Approach (a)
- Input window size varies with latency (iW = L×16)
- Output window size varies with latency (oW = L×16)

### Approach (b)
- Fixed input window size (iW = 256 samples = 16ms)
- Output window size varies with latency (oW = L×16)

## Features

- **Low Latency**: Supports latencies from 1ms to 16ms
- **Lightweight**: Models range from ~100k to ~25M parameters
- **Multichannel**: Supports 2, 4, and 8 channel configurations
- **Time Domain**: Operates directly on time-domain signals
- **Efficient**: Optimized for real-time processing on edge devices

## Technical Details

### Spatial Filtering

The model uses learnable spatial filters to combine information across multiple input channels:

```python
self.spatial_filters = nn.Parameter(torch.empty(hidden_dim, n_channels))
nn.init.kaiming_uniform_(self.spatial_filters)
```

### Overlap-Add Processing

The model uses an overlap-add approach with configurable stride for efficient processing of long audio signals.

### Normalization

Layer normalization is applied before each LSTM layer to stabilize training and improve convergence.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{pandey2023simple,
  title={A Simple RNN Model for Lightweight, Low-compute and Low-latency Multichannel Speech Enhancement in the Time Domain},
  author={Pandey, Ashutosh and Tan, Ke and Xu, Buye},
  booktitle={Proc. Interspeech 2023},
  pages={2478--2482},
  year={2023}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Acknowledgments

This is an unofficial implementation of the paper. Please refer to the original paper for the official implementation and more details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
