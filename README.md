# Digital Twin-Aided Channel Estimation

Implementation of zone-specific subspace prediction and calibration for wireless channel estimation using digital twin models.

## 📋 Overview

This repository implements the research presented in the paper **"Digital twin aided channel estimation: Zone-specific subspace prediction and calibration"** ([arXiv:2501.02758](https://doi.org/10.48550/arXiv.2501.02758)). The system provides two distinct approaches for digital twin-aided channel estimation using different datasets and methodologies.

## 🎯 Two Implementation Approaches

### Part 1: DeepMIMO Indianapolis Dataset Approach
**File: `main.py`**

This approach uses the original Indianapolis datasets from DeepMIMO with building shifts and lower ray-tracing fidelity for digital twin generation.

#### Key Features
- **Real-World Dataset**: Original Indianapolis 28GHz dataset with full ray-tracing fidelity
- **Digital Twin Dataset**: Indianapolis dataset with shifted buildings and reduced ray-tracing fidelity
- **Building Shift Strategy**: DT buildings are shifted to create simplified channel models
- **Lower Fidelity**: Reduced ray-tracing complexity for computational efficiency

#### Usage
```bash
python main.py
```

#### Configuration
```python
# DeepMIMO Indianapolis scenarios
scenarios = ['indianapolis_original_28GHz', 'indianapolis_2mShifted_28GHz', 'indianapolis_4mShifted_28GHz']
n_beams = 64
n_path = [10, 20]  # [DT paths, RW paths]
```

#### Dataset Structure
```
scenarios/
├── indianapolis_original_28GHz/     # Real-World dataset
├── indianapolis_2mShifted_28GHz/   # DT dataset (2m building shift)
└── indianapolis_4mShifted_28GHz/   # DT dataset (4m building shift)
```

---

### Part 2: DeepVerse Carla-Town05 Dataset Approach
**File: `main_deepverse.py`**

This approach uses the Carla-Town05 dataset from DeepVerse with Doppler effects and lower ray-tracing fidelity for digital twin generation.

#### Key Features
- **Real-World Dataset**: Carla-Town05 with Doppler effects enabled and full ray-tracing fidelity
- **Digital Twin Dataset**: Carla-Town05 with Doppler disabled and reduced ray-tracing fidelity
- **Doppler Control**: RW has Doppler enabled, DT has Doppler disabled
- **Lower Fidelity**: Reduced ray-tracing complexity for computational efficiency

#### Usage

**Local Execution:**
```bash
python main_deepverse.py
```

**Interactive Jupyter Notebook (Google Colab):**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadjadalikhani/Digital-twin-aided-channel-estimation/blob/main/main_deepverse.ipynb)

Try the interactive notebook version with pre-configured environment and visualizations!

#### Configuration
```python
# DeepVerse Carla-Town05 scenarios
scenarios = np.arange(10)        # Number of scenarios to use
n_beams = 128                    # Number of antenna beams
n_path = [5, 25]                 # [DT paths, RW paths]
trials = 20                      # Number of simulation trials
```

#### Dataset Structure
```
scenarios/
└── Carla-Town05/
    ├── param/                   # Configuration parameters
    └── wireless/               # Channel data
        ├── scene_0/           # Individual scene data
        ├── scene_1/
        └── ...
```

## 🏗️ Architecture

### Core Components

```
├── main.py                      # DeepMIMO Indianapolis approach
├── main_deepverse.py            # DeepVerse Carla-Town05 approach
├── deepverse/                   # DeepVerse integration package
│   ├── deepverse_comm.py       # Communication channel utilities
│   └── deepverse_dt_rw_channel_gen.py  # DT/RW channel generation
├── utils.py                     # Core algorithms and utilities
├── input_preprocess.py          # Data preprocessing and visualization
└── scenarios/                   # Dataset scenarios
    ├── indianapolis_*/         # DeepMIMO datasets
    └── Carla-Town05/           # DeepVerse dataset
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda environment with required packages
- DeepMIMO and DeepVerse simulation frameworks
- PyTorch, NumPy, Matplotlib, Scikit-learn

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Digital-twin-aided-channel-estimation.git
   cd Digital-twin-aided-channel-estimation
   ```

2. **Set up the environment**:
   ```bash
   conda create -n dtce python=3.8
   conda activate dtce
   pip install torch numpy matplotlib scikit-learn
   ```

3. **Choose your approach**:
   ```bash
   # For DeepMIMO Indianapolis approach
   python main.py
   
   # For DeepVerse Carla-Town05 approach
   python main_deepverse.py
   ```

## 📊 Results and Visualization

Both approaches generate comprehensive results including:

- **Channel Estimation Performance**: NMSE, cosine similarity, and throughput metrics
- **Zone Analysis**: Spatial clustering and subspace characteristics
- **Comparative Studies**: Digital Twin vs. Real-World performance analysis
- **Pilot Optimization**: Performance vs. number of pilots analysis

All figures are automatically saved to the `figs/` directory with high-resolution output.

## 🔬 Research Applications

### Digital Twin Methodology

- **Simplified Models**: Reduced complexity models for computational efficiency
- **Real-World Calibration**: Full complexity models for realistic simulation
- **Perfect Pairing**: Ensures one-to-one correspondence between DT and RW channels

### Machine Learning Integration

- **Subspace Learning**: Principal component analysis for channel subspace identification
- **Zone-Specific Training**: Spatial clustering for localized channel estimation
- **Deep Reinforcement Learning**: DRL-based pilot selection optimization

### Performance Metrics

- **Normalized Mean Square Error (NMSE)**: Channel estimation accuracy
- **Cosine Similarity**: Directional channel correlation
- **Throughput Analysis**: System-level performance evaluation

## 🛠️ Advanced Usage

### Custom Channel Generation (DeepVerse Approach)

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

## 📜 License

MIT License - see LICENSE file for details.