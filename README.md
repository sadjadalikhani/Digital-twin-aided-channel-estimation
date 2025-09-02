# Digital Twin-Aided Channel Estimation

Effective channel estimation in sparse and high-dimensional environments is essential for next-generation wireless systems, particularly in large-scale MIMO deployments. This paper introduces a novel framework that leverages digital twins (DTs) as priors to enable efficient zone-specific subspace-based channel estimation (CE). Subspace-based CE significantly reduces feedback overhead by focusing on the dominant channel components, exploiting sparsity in the angular domain while preserving estimation accuracy. While DT channels may exhibit inaccuracies, their coarse-grained subspaces provide a powerful starting point, reducing the search space and accelerating convergence. The framework employs a two-step clustering process on the Grassmann manifold, combined with reinforcement learning (RL), to iteratively calibrate subspaces and align them with real-world counterparts. Simulations show that digital twins not only enable near-optimal performance but also enhance the accuracy of subspace calibration through RL, highlighting their potential as a step towards learnable digital twins.

## 📋 Overview

This repository implements the research presented in the paper **"Digital twin aided channel estimation: Zone-specific subspace prediction and calibration"** ([arXiv:2501.02758](https://doi.org/10.48550/arXiv.2501.02758)). The system leverages digital twin technology to create simplified channel models that can be used for efficient channel estimation in real-world wireless environments.

### Key Features

- **Digital Twin Channel Generation**: Simplified channel models with reduced complexity (5 paths, no Doppler)
- **Real-World Channel Simulation**: Comprehensive channel models with full complexity (25 paths, Doppler effects)
- **Zone-Specific Subspace Estimation**: K-means clustering and subspace-based channel estimation
- **DeepVerse Integration**: Advanced wireless channel simulation using DeepVerse framework
- **Machine Learning Pipeline**: Support for various estimation algorithms including DRL-based optimization
- **Performance Evaluation**: Comprehensive metrics including NMSE, cosine similarity, and throughput analysis

## 🏗️ Architecture

### Core Components

```
├── main_deepverse.py              # Main simulation and training pipeline
├── deepverse/                     # DeepVerse integration package
│   ├── deepverse_comm.py         # Communication channel utilities
│   └── deepverse_dt_rw_channel_gen.py  # DT/RW channel generation
├── utils.py                      # Core algorithms and utilities
├── input_preprocess.py           # Data preprocessing and visualization
└── scenarios/                    # DeepVerse scenario data
    └── Carla-Town05/            # Urban environment scenarios
```

### Data Flow

1. **Channel Generation**: Generate paired Digital Twin and Real-World channel datasets
2. **User Overlay**: Combine users from multiple scenes for enhanced ML training
3. **Zone Clustering**: K-means clustering for spatial zone identification
4. **Subspace Estimation**: Zone-specific subspace learning and calibration
5. **Performance Evaluation**: Comprehensive analysis across multiple metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda environment with required packages
- DeepVerse simulation framework
- PyTorch, NumPy, Matplotlib, Scikit-learn

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Digital-twin-aided-channel-estimation.git
   cd Digital-twin-aided-channel-estimation
   ```

2. **Set up the environment**:
   ```bash
   conda create -n lwmv2 python=3.8
   conda activate lwmv2
   pip install -r requirements.txt  # If available
   ```

3. **Run the main simulation**:
   ```bash
   python main_deepverse.py
   ```

### Configuration

Key parameters can be adjusted in `main_deepverse.py`:

```python
scenarios = np.arange(10)        # Number of scenarios to use
n_beams = 128                    # Number of antenna beams
n_path = [5, 25]                 # [DT paths, RW paths]
trials = 20                      # Number of simulation trials
```

## 📊 Results and Visualization

The system generates comprehensive results including:

- **Channel Estimation Performance**: NMSE, cosine similarity, and throughput metrics
- **Zone Analysis**: Spatial clustering and subspace characteristics
- **Comparative Studies**: Digital Twin vs. Real-World performance analysis
- **Pilot Optimization**: Performance vs. number of pilots analysis

All figures are automatically saved to the `figs/` directory with high-resolution output.

## 🔬 Research Applications

### Digital Twin Methodology

- **Simplified Models**: 5-path channels without Doppler effects for computational efficiency
- **Real-World Calibration**: 25-path channels with full Doppler for realistic simulation
- **Perfect Pairing**: Ensures one-to-one correspondence between DT and RW channels

### Machine Learning Integration

- **Subspace Learning**: Principal component analysis for channel subspace identification
- **Zone-Specific Training**: Spatial clustering for localized channel estimation
- **Deep Reinforcement Learning**: DRL-based pilot selection optimization

### Performance Metrics

- **Normalized Mean Square Error (NMSE)**: Channel estimation accuracy
- **Cosine Similarity**: Directional channel correlation
- **Throughput Analysis**: System-level performance evaluation

## 📁 Dataset Structure

The system works with DeepVerse-generated scenarios:

```
scenarios/
└── Carla-Town05/
    ├── param/                   # Configuration parameters
    └── wireless/               # Channel data
        ├── scene_0/           # Individual scene data
        ├── scene_1/
        └── ...
```

## 🛠️ Advanced Usage

### Custom Channel Generation

```python
from deepverse.deepverse_dt_rw_channel_gen import chs_gen

# Generate paired DT/RW datasets
dataset_dt, dataset_rw, pos, los_status, best_beam, enabled_idxs, bs_pos = chs_gen(
    scenarios=scenarios,
    n_beams=128,
    fov=180,
    n_path=[5, 25],
    codebook=codebook
)
```

### Zone-Specific Analysis

```python
from utils import k_means, k_med

# Perform spatial clustering
dt_subspaces, rw_subspaces, kmeans_centroids, kmeans_labels = k_means(
    enabled_idxs, dataset_dt, dataset_rw, pos, los_status, 
    best_beam, bs_pos, pos_coeff, los_coeff_kmeans, beam_coeff_kmeans
)
```

## 📈 Performance Optimization

- **Adaptive Clustering**: Dynamic adjustment of cluster numbers based on dataset size
- **Memory Management**: Efficient tensor operations with PyTorch
- **Parallel Processing**: Support for multi-scenario processing
- **Figure Generation**: Automatic saving without display interruption

## 🤝 Contributing

We welcome contributions to improve the digital twin channel estimation framework. Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{alikhani2025digitaltwinaidedchannel,
      title={Digital Twin Aided Channel Estimation: Zone-Specific Subspace Prediction and Calibration}, 
      author={Sadjad Alikhani and Ahmed Alkhateeb},
      year={2025},
      eprint={2501.02758},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2501.02758}, 
}
```
