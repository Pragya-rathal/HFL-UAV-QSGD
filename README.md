# 🚁 UAV-Assisted Hierarchical Federated Learning (HFL) with Compression

A research-grade simulation framework for **Hierarchical Federated Learning (HFL)** in UAV-assisted IoT networks, with integrated **gradient compression (Top-K, QSGD)** and **quorum-based device selection**.

---

## 📌 Overview

This project simulates a **multi-tier federated learning system**:

- 📱 IoT devices perform local training  
- 🧩 Devices are grouped via clustering  
- 🚁 UAVs act as intermediate aggregators  
- 🌍 A global model is updated iteratively  

The framework evaluates:

- Model accuracy  
- Communication overhead  
- Training latency  
- Device participation  

---

## 🧠 Key Features

- Standard Federated Learning (FedAvg)
- Hierarchical / Clustered FL
- Top-K Gradient Compression + Error Feedback
- QSGD Quantization
- Quorum-based device selection
- FedProx (proximal regularisation baseline)
- TopoCo: topology-adaptive primal-dual co-optimisation
- IID and Non-IID (Dirichlet) data splits
- Full experiment + plotting pipeline

---

## 📂 Project Structure

```
.
├── main.py              # Entry point
├── federated.py         # FL algorithms (6 classic methods + dispatch)
├── federated_round.py   # TopoCo per-round primal-dual loop
├── baselines.py         # FedProx baseline
├── model.py             # CNN models (MNIST / CIFAR-10)
├── compression.py       # Top-K + QSGD
├── clustering.py        # Cluster formation + head election
├── data_loader.py       # MNIST / CIFAR-10 loader
├── metrics.py           # Metrics + summary
├── plotting.py          # Base visualizations
├── plotting_topoco.py   # TopoCo-specific visualizations
├── topology.py          # Dynamic communication graph
├── utility.py           # Lagrangian state + dual updates
├── schedules.py         # Adaptive compression + participation
├── config.py            # Experiment configs
├── devices.py           # Device simulation
└── results/             # Outputs
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Pragya-rathal/uav-project-QSGD.git
cd uav-project-QSGD
```

---

## 🧪 Running Experiments

### 🔹 Toy Mode (MNIST)

```bash
python main.py --mode toy
```

### 🔹 Full Mode (CIFAR-10)

```bash
python main.py --mode full
```

---

## 📊 Methods Implemented

| Method | Description |
|--------|-------------|
| A | Standard FL (FedAvg) |
| B | Clustered FL |
| C | Top-K + Error Feedback |
| D | QSGD |
| E | Top-K + Quorum |
| F | QSGD + Quorum |
| G | FedProx (modern baseline) |
| H | TopoCo — primal-dual topology-adaptive co-optimisation (ours) |

---

## 📈 Outputs

Generated automatically in `results/`:

- Accuracy vs Rounds
- Loss vs Rounds
- Latency vs Rounds
- Communication vs Rounds
- Tradeoff plots (Accuracy vs Latency / Comm)
- Cluster latency distributions

---

## 📡 System Model

### IoT Devices
- Compute power
- Bandwidth
- Distance (affects latency)
- Local dataset size

### Clustering
- Distance-aware grouping
- Cluster-head selection

### Compression

**Top-K + Error Feedback**
- Sends only largest gradients
- Uses residual correction

**QSGD**
- Stochastic quantization
- Reduces communication cost

---

## 📚 Datasets

- MNIST (toy mode)
- CIFAR-10 (full mode)

Supports:
- IID splits
- Non-IID Dirichlet splits

---

## 🔬 Research Focus

- Communication vs accuracy trade-offs  
- Impact of hierarchical aggregation  
- Compression efficiency  
- Latency-aware FL  

---

## ⚠️ Notes

- GPU optional (CPU works fine)
- Fully reproducible via seeds
- Data auto-downloads (or falls back to synthetic data)

---

## 🧾 Citation

```bibtex
@article{uav_hfl_qsgd_2026,
  title={Hierarchical Federated Learning in UAV-Assisted IoT Networks with Compression},
  author={Pragya Rathal},
  journal={IEEE Transactions (Target)},
  year={2026}
}
```
